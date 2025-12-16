######################################################################################################################################################
#
#   name:       STAC_Utils.py
#   contains:   Functions for accessing STAC data (usually as xarray)
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Built in
import os
import datetime as dt
import warnings as warn
import calendar as cal
import time
import traceback

from dateutil.rrule import rrule, MONTHLY, DAILY

# Open source
import pystac_client as pc
import stackstac as ss
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import cupy as cp
import polars as pl
import rioxarray # Used in rio.clip

from rasterio.enums import Resampling
#from retry import retry

# Mine
import PreProcess_Utils as pputil

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Consider odc-stac (vs stackstac) - seemed a bit slower in limited testing but more active development. Has code for dask-based cloud masking. 
# dask.delayed on top of process like observationAvailabilityHLS() may be better than for loop (https://docs.dask.org/en/stable/delayed.html)
# - https://discourse.pangeo.io/t/compute-time-series-for-70-000-locations-speed-up-the-processing/4436/7 (further discussion)
# https://discourse.pangeo.io/t/hls-time-series-using-xarray-best-practices/4578 (odc-stac recommended over stackstac here)
# retry-after may help avoid manual code restart after STAC API errors (https://github.com/nasa/cmr-stac/issues/386#issuecomment-2625251186)
# Save data cubes in a better format than current NetCDF approach (e.g., Zarr + Icechunk)?

######################################################################################################################################################
# Build Cubes ########################################################################################################################################
######################################################################################################################################################

# Builds HLS spectral and/or Fmask cubes as Dask-backed Lazy Arrays.
#@retry()
def buildHLS(area, 
             start = '2013-01-01', 
             end = dt.datetime.today().strftime('%Y-%m-%d'), 
             bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'Fmask'],
             chunksize = (1, -1, -1, -1), 
             sceneCloudThresh = 90, 
             proj = 'UTM',
             accessEarthData = True,
             gdalEnv = None,
             catalog = None,
             verbose = False):

    """
    Parameters: 
    area (string or gdf): path to shp ('C:/path/to/poly.shp') or GeoDataFrame

    start (string): When to start HLS data cube ('YYYY-MM-DD')

    end (string): When to end HLS data cube ('YYYY-MM-DD')

    bands (list): Band names of interest. Supports: 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'Fmask'. 
    - If spectral and Fmask suppled, builds a spectral cube and a mask cube
    - If only spectral supplied, builds a spectral cube
    - If only Fmask supplied, builds a mask cube
    - Bands shpuld be ordered from BLUE to SWIR2, and Fmask should be final band in list

    chunksize (tuple of ints or 'auto'): chunksize of Dasked-back xarray in dimensions (time, band, x, y). 
    - Default: One timestep, all bands, all spatial (good for smaller area, deep in time processing).
    - For larger area, shallow in time processing: may need to use different spatial values (e.g., 5000 by 5000 pixels).
    - 'auto': Let stackstac decide. 

    sceneCloudThresh (int): Highest scene level cloud % to include 

    proj (string): Projection of output cube. 'UTM' means output is in local UTM grid. 'poly' means projection matches area input. 

    accessEarthData (boolean): Whether (true) or not (false) to access NASA Earthdata STAC within function.
    - If running in for loop (e.g., over tiles), set to False and run accessSTAC() just once before to save time.

    gdalEnv (dict): GDAL environment settings. Required if accessSTAC is False.

    catalog (pystac_client.Catalog): Catalog object. Required if accessSTAC is False.

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube (in-memory xarray): Pre-processed HLS cube with smallest datasize (e.g., int16 for spectral, uint8 for fmask).
    """
    # Get bboxes based on area
    bboxLL = pputil.poly2bbox(area, 'lat/lon') # For STAC search
    bbox, epsg = pputil.poly2bbox(area, proj) # For stackstac

    if verbose == True:
        print('Bounding boxes created and projection defined (EPSG:' + str(epsg) + ').')

    # Access NASA Earthdata STAC
    if accessEarthData == True:
        gdalEnv, catalog = accessSTAC('https://cmr.earthdata.nasa.gov/stac/LPCLOUD') # 1-2 seconds overhead here if running buildHLS in for loop

        if verbose == True:
            print('Connected to NASA Earthdata (LPCLOUD).')

    # Get all L30 items (Default limit is slower, faster leads to errors) #'HLSL30.v2.0'
    itemsL30 = catalog.search(bbox = bboxLL, datetime = f'{start}/{end}', collections = ['HLSL30_2.0'], limit = 100).item_collection() 
    nL30 = len(itemsL30)

    if verbose == True: 
        print('Nearby L30 images found on NASA EarthData (n = ' + str(nL30) + ').')

    # Get all S30 items # HLSS30.v2.0'
    itemsS30 = catalog.search(bbox = bboxLL, datetime = f'{start}/{end}', collections = ['HLSS30_2.0'], limit = 100).item_collection() 
    nS30 = len(itemsS30)    

    if verbose == True:
        print('Nearby S30 images found on NASA EarthData (n = ' + str(nS30) + ').')   

    if (nL30 == 0) & (nS30 == 0):
        raise Exception('No HLS scenes for selected location and time.')   

    # Select and align spectral band names (outputs empty list if just getting Fmask)
    bandsL30, bandsS30 = pputil.bandBuilder(bands, 'hls')

    # L30
    if nL30 > 0:  # Need to have at least 1 image

        # Build L30 spectral cube
        if 'BLUE' in bands or 'GREEN' in bands or 'RED' in bands or 'NIR' in bands or 'SWIR1' in bands or 'SWIR2' in bands:

            if verbose == True:
                print('...Building L30 spectral cube...')

            with warn.catch_warnings():
                warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac

                l30s = ss.stack(itemsL30, 
                                assets = bandsL30, 
                                epsg = epsg, 
                                resolution = 30, 
                                bounds = bbox, 
                                resampling = Resampling.cubic, # continuous
                                chunksize = chunksize, # 1 time-step, all bands, full extent
                                xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                                dtype = 'int16', # Reduces size to 25% of default (float64)
                                fill_value = np.uint16(-9999), # -9999 # Specifying np.dtype required for ss 0.5.1
                                rescale = False, 
                                gdal_env = gdalEnv)  

                if verbose == True:
                    print('Removed images not touching bounding box (n = ' + str(l30s.shape[0]) + ').') 

            # Remove bad time-steps from cube based on scene-level metadata
            l30s = pputil.removeBadScenes(l30s, sceneCloudThresh, verbose)

            # Merge images from observations on the same day
            l30s = pputil.sameDayMerge(l30s, 'spectral', 'L30', bands, 'hls', verbose)

        # Build L30 mask cube
        if 'Fmask' in bands:

            if verbose == True:
                print('...Building L30 Fmask cube...')

            with warn.catch_warnings():
                warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac

                l30m = ss.stack(itemsL30, 
                                assets = ['Fmask'], 
                                epsg = epsg, 
                                resolution = 30, 
                                bounds = bbox, 
                                resampling = Resampling.nearest, # discrete
                                chunksize = chunksize, # 1 time-step, all bands, full extent
                                xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                                dtype = 'uint8', # Reduces size to 12.5% of default (float64)
                                fill_value = np.uint8(255), # 255 # Specifying np.dtype required for ss 0.5.1
                                rescale =  False,
                                gdal_env = gdalEnv)
                
                if verbose == True:
                    print('Removed images not touching bounding box (n = ' + str(l30m.shape[0]) + ').') 
                
            # Remove bad time-steps from cube based on scene-level metadata
            l30m = pputil.removeBadScenes(l30m, sceneCloudThresh, verbose)

            # Merge images from observations on the same day
            l30m = pputil.sameDayMerge(l30m, 'mask', 'L30', bands, 'hls', verbose)            
                
    # S30
    if nS30 > 0:  # Need to have at least 1 image

        # Build S30 spectral cube
        if 'BLUE' in bands or 'GREEN' in bands or 'RED' in bands or 'NIR' in bands or 'SWIR1' in bands or 'SWIR2' in bands:

            if verbose == True:
                print('...Building S30 spectral cube...')

            with warn.catch_warnings():
                warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac

                s30s = ss.stack(itemsS30, 
                                assets = bandsS30, 
                                epsg = epsg, 
                                resolution = 30, 
                                bounds = bbox, 
                                resampling = Resampling.cubic, # continuous
                                chunksize = chunksize, # 1 time-step, all bands, full exten
                                xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                                dtype = 'int16', # Reduces size to 25% of default (float64)
                                fill_value = np.uint16(-9999), # -9999 # Specifying np.dtype required for ss 0.5.1
                                rescale = False,
                                gdal_env = gdalEnv)
                
                if verbose == True:
                    print('Removed images not touching bounding box (n = ' + str(s30s.shape[0]) + ').') 
                
            # Remove bad time-steps from cube based on scene-level metadata
            s30s = pputil.removeBadScenes(s30s, sceneCloudThresh, verbose)
 
            # Merge images from observations on the same day
            s30s = pputil.sameDayMerge(s30s, 'spectral', 'S30', bands, 'hls', verbose)

        # Build S30 mask cube
        if 'Fmask' in bands:

            if verbose == True:
                print('...Building S30 Fmask cube...')

            with warn.catch_warnings():
                warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac

                s30m = ss.stack(itemsS30, 
                                assets = ['Fmask'], 
                                epsg = epsg, 
                                resolution = 30, 
                                bounds = bbox, 
                                resampling = Resampling.nearest, # discrete
                                chunksize = chunksize, # 1 time-step, all bands, full extent
                                xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                                dtype = 'uint8', # Reduces size to 12.5% of default (float64)
                                fill_value = np.uint8(255), # 255 # Specifying np.dtype required for ss 0.5.1
                                rescale =  False,
                                gdal_env = gdalEnv)
                
                if verbose == True:
                    print('Removed images not touching bounding box (n = ' + str(s30m.shape[0]) + ').') 
                
            # Remove bad time-steps from cube based on scene-level metadata
            s30m = pputil.removeBadScenes(s30m, sceneCloudThresh, verbose)

            # Merge images from observations on the same day
            s30m = pputil.sameDayMerge(s30m, 'mask', 'S30', bands, 'hls', verbose)     

    # Combine into HLS cubes
    # If both L30 and S30 have images...
    if (nL30 > 0) & (nS30 > 0):

        # Combine into HLS spectral cube
        if 'BLUE' in bands or 'GREEN' in bands or 'RED' in bands or 'NIR' in bands or 'SWIR1' in bands or 'SWIR2' in bands:
            hlss = xr.concat((l30s, s30s), dim = 'time').sortby('time')

            if verbose == True:
                print('Combined L30 and S30 into HLS spectral cube (n = ' + str(hlss.shape[0]) + ').')

        # Combine into HLS Fmask cube
        if 'Fmask' in bands:
            hlsm = xr.concat((l30m, s30m), dim = 'time').sortby('time')  

            if verbose == True:
                print('Combined L30 and S30 into HLS Fmask cube (n = ' + str(hlsm.shape[0]) + ').')    

    # If only L30 has images...
    if (nL30 > 0) & (nS30 == 0):

        # Combine into HLS spectral cube
        if 'BLUE' in bands or 'GREEN' in bands or 'RED' in bands or 'NIR' in bands or 'SWIR1' in bands or 'SWIR2' in bands:
            hlss = l30s

        # Combine into HLS Fmask cube
        if 'Fmask' in bands:
            hlsm = l30m

        if verbose == True:
            print('No combining, L30 only cube.')
        
     # If only S30 has images...
    if (nS30 > 0) & (nL30 == 0):

        # Combine into HLS spectral cube
        if 'BLUE' in bands or 'GREEN' in bands or 'RED' in bands or 'NIR' in bands or 'SWIR1' in bands or 'SWIR2' in bands:        
            hlss = s30s

        # Combine into HLS Fmask cube
        if 'Fmask' in bands:
            hlsm = s30m

        if verbose == True:
            print('No combining, S30 only cube.')

    # Save outputs as variables
    if 'Fmask' in bands:
        if len(bands) == 1: # Just Fmask
            return hlsm
        if len(bands) > 1: # Fmask + spectral bands
            return hlss, hlsm     
    else:
        return hlss
######################################################################################################################################################
    
# Builds S2 spectral and (if asked for) SCL cubes as Dask-backed Lazy Arrays.
#@retry()
def buildS2(area, 
             start = '2017-01-01', 
             end = dt.datetime.today().strftime('%Y-%m-%d'), 
             bands = ['BLUE', 'GREEN', 'RED', 'REDEDGE1', 'REDEDGE2', 'REDEDGE3', 'NIR0', 'NIR', 'SWIR1', 'SWIR2', 'SCL'], 
             chunksize = (1, -1, -1, -1), 
             sceneCloudThresh = 90, 
             proj = 'UTM',
             verbose = False):

    """
    Parameters: 
    area (string): path to shp ('C:/path/to/poly.shp') or GeoDataFrame

    start (string): When to start S2 data cube ('YYYY-MM-DD')

    end (string): When to end S2 data cube ('YYYY-MM-DD')

    bands (list): Band names of interest. Supports: 'BLUE', 'GREEN', 'RED', 'REDEDGE1', 'REDEDGE2', 'REDEDGE3', 'NIR0', 'NIR', 'SWIR1', 'SWIR2', 
    'SCL'
    - If spectral and SCL suppled, builds a spectral cube and a mask cube
    - If only spectral supplied, builds a spectral cube
    - If only SCL supplied, builds a mask cube
    - Bands shpuld be ordered from BLUE to SWIR2, and SCL should be final band in list    

    chunksize (tuple of ints or 'auto'): chunksize of Dasked-back xarray in dimensions (time, band, x, y). 
    - Default: One timestep, all bands, all spatial (good for smaller area, deep in time processing).
    - For larger area, shallow in time processing: may need to use different spatial values (e.g., 5000 by 5000 pixels).
    - 'auto': Let stackstac decide

    sceneCloudThresh (int): Highest scene level cloud % to include 

    proj (string): Projection of output cube. 'UTM' means output is in local UTM grid. 'poly' means projection matches area input. 

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube (in-memory xarray): Pre-processed S2 cube  with smallest datasize (e.g., int16 for spectral, uint8 for SCL).
    """
    # Get bboxes based on area
    bboxLL = pputil.poly2bbox(area, 'lat/lon') # For STAC search
    bbox, epsg = pputil.poly2bbox(area, proj) # For stackstac

    # Access AWS E84 STAC
    gdalEnv, catalog = accessSTAC('https://earth-search.aws.element84.com/v1')

    if verbose == True:
        print('Connected to AWS E84.')

    # Get all S2 items
    items = catalog.search(bbox = bboxLL, datetime = f'{start}/{end}', collections = ['sentinel-2-l2a'], limit = 100).item_collection()
    nItems = len(items)

    if verbose == True: 
        print('S2 images found on AWS E84 (n = ' + str(nItems) + ').')

    if nItems == 0:
        raise Exception('No S2 L2A images for selected location & time.') 

    # Select and align spectral band names (outputs empty list if just getting SCL)
    bands, bandsS2 = pputil.bandBuilder(bands, 's2')

    # Build S2 spectral cube
    if 'BLUE' in bands or 'GREEN' in bands or 'RED' in bands or 'REDEDGE1' in bands or 'REDEDGE2' in bands or 'REDEDGE3' in bands \
    or 'NIR0' in bands or 'NIR' in bands or 'SWIR1' in bands or 'SWIR2' in bands:
        
        if verbose == True:
            print('...Building S2 spectral cube...')

        with warn.catch_warnings():
            warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac            

            # Build initial S2 cube
            s2s = ss.stack(items, 
                           assets = bandsS2, 
                           epsg = epsg,
                           resolution = 10, 
                           bounds = bbox, 
                           chunksize = chunksize, # 1 time-step, all bands, full extent
                           xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                           dtype = 'uint16', # Reduces size to 25% of default (float64)
                           fill_value = 0,
                           rescale = False,
                           gdal_env = gdalEnv) 

        # Remove bad time-steps from cube based on scene-level metadata
        s2s = pputil.removeBadScenes(s2s, sceneCloudThresh, verbose)

        # Merge images from observations on the same day
        s2s = pputil.sameDayMerge(s2s, 'spectral', 'S2', bands, 's2', verbose)   

    # Build S2 mask cube
    if 'SCL' in bands:  

        if verbose == True:
            print('...Building S2 SCL cube...')

        with warn.catch_warnings():
            warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac   

            s2m = ss.stack(items, 
                           assets = ['scl'], 
                           epsg = epsg, 
                           resolution = 10, 
                           bounds = bbox, 
                           resampling = Resampling.nearest, # discrete
                           chunksize = chunksize, # 1 time-step, all bands, full extent
                           xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                           dtype = 'uint8', # Reduces size to 12.5% of default (float64)
                           fill_value = 0,
                           rescale =  False,
                           gdal_env = gdalEnv)        

            # Remove bad time-steps from cube based on scene-level metadata
            s2m = pputil.removeBadScenes(s2m, sceneCloudThresh, verbose)

            # Mergeimages from observations on the same day
            s2m = pputil.sameDayMerge(s2m, 'mask', 'S2', bands, 's2', verbose)               

    # Save outputs as variables
    if 'SCL' in bands:
        if len(bands) == 1: # Just SCL
            return s2m
        if len(bands) > 1: # SCL + spectral bands
            return s2s, s2m
    else:
        return s2s

######################################################################################################################################################
# Assess HLS #########################################################################################################################################
######################################################################################################################################################

# Get HLS observation availability and other metadata for given area(s).
def observationAvailabilityHLS(area_shp, 
                               csv, 
                               year, 
                               proj = 'poly', 
                               progressTrack = 'Complete',
                               borderCleaning = '', 
                               roi_shp = '', 
                               download = '', 
                               downloadPath = '', 
                               metadata = True, 
                               verbose = True):

    """
    Parameters:
    area_shp (str): Path to shp ('C:/path/to/poly.shp'). Area with polygons to be processed. 

    csv (str): Path to directory where csvs are stored ('C:/path/to/csvs'). Will look for folders named 'Metadata' and 'Observations'. 

    year (int): Year to gather data from.

    proj (str): Projection of output cube. 'UTM' means output is in local UTM grid. 'poly' means projection matches area input.

    progressTrack (str): Column name in area to track progress (should be all 0s to start). Will change value to 1 when polygon completed. 
    - Must be provided since this is used to restart from where left off previously on new run

    borderCleaning (str): Column name in area with % (0 - 100) coverage of polygon within region of interest (roi). 
    - If '', do not implement border cleaning.
    - Use something like Tabulate Intersection in ArcGIS to get overlap %

    roi_shp (str): Path to shp ('C:/path/to/poly.shp'). Area representing region of interest. Works with borderCleaning. 
    - Pixels outside roi will be masked (value 0)

    download (str): Column name in area indicating which polygons to download fmask data from as NetCDF. 1 = download, 0 = do not download.

    downloadPath (str): Folder to save netCDF files for download polygons ('C:/path/to/file').

    metadata (bool): Whether (true) or not (false) to record metadata (processing time, size etc.)

    verbose (bool): Whether (true) or not (false) to print function status.

    Returns:
    CSVs with observation availability information and other metadata
    """

    # Load area as geodataframe
    area = gpd.read_file(area_shp, engine = 'pyogrio')
    
    # Load CSVs to Fill
    # Schema (float32 for polygon columns)
    if metadata == False:
        obs_schema = OAHLSschema(area, metadata)
    if metadata == True:
        obs_schema, meta_schema = OAHLSschema(area, metadata)
        meta = pl.read_csv(csv + '/Metadata/meta_' + str(year) + '.csv', schema = meta_schema) 

    # Observations
    obs_land_L30 = pl.read_csv(csv + '/Observations/obs_land_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_land_S30 = pl.read_csv(csv + '/Observations/obs_land_S30_' + str(year) + '.csv', schema = obs_schema)
    obs_water_L30 = pl.read_csv(csv + '/Observations/obs_water_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_water_S30 = pl.read_csv(csv + '/Observations/obs_water_S30_' + str(year) + '.csv', schema = obs_schema)
    obs_snow_L30 = pl.read_csv(csv + '/Observations/obs_snow_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_snow_S30 = pl.read_csv(csv + '/Observations/obs_snow_S30_' + str(year) + '.csv', schema = obs_schema)
    obs_aerosolL_L30 = pl.read_csv(csv + '/Observations/obs_aerosolL_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_aerosolL_S30 = pl.read_csv(csv + '/Observations/obs_aerosolL_S30_' + str(year) + '.csv', schema = obs_schema)    
    obs_aerosolW_L30 = pl.read_csv(csv + '/Observations/obs_aerosolW_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_aerosolW_S30 = pl.read_csv(csv + '/Observations/obs_aerosolW_S30_' + str(year) + '.csv', schema = obs_schema)
    obs_aerosolS_L30 = pl.read_csv(csv + '/Observations/obs_aerosolS_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_aerosolS_S30 = pl.read_csv(csv + '/Observations/obs_aerosolS_S30_' + str(year) + '.csv', schema = obs_schema)    
    obs_adjacent_L30 = pl.read_csv(csv + '/Observations/obs_adjacent_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_adjacent_S30 = pl.read_csv(csv + '/Observations/obs_adjacent_S30_' + str(year) + '.csv', schema = obs_schema)
    obs_shadow_L30 = pl.read_csv(csv + '/Observations/obs_shadow_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_shadow_S30 = pl.read_csv(csv + '/Observations/obs_shadow_S30_' + str(year) + '.csv', schema = obs_schema)  
    obs_cloud_L30 = pl.read_csv(csv + '/Observations/obs_cloud_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_cloud_S30 = pl.read_csv(csv + '/Observations/obs_cloud_S30_' + str(year) + '.csv', schema = obs_schema)
    obs_fill_L30 = pl.read_csv(csv + '/Observations/obs_fill_L30_' + str(year) + '.csv', schema = obs_schema)
    obs_fill_S30 = pl.read_csv(csv + '/Observations/obs_fill_S30_' + str(year) + '.csv', schema = obs_schema) 
        
    if verbose == True:
        #print('Loaded CSVs with', ((obs_fill_L30.null_count() < len(obs_fill_L30)).sum_horizontal() - 1).item(), 'completed polygons.')
        print('Loaded CSVs with', round(area[progressTrack].sum()), 'completed polygons.')

    # roi
    if borderCleaning != '':
        roi = gpd.read_file(roi_shp, engine = 'pyogrio')

    # Access NASA Earthdata STAC
    gdalEnv, catalog = accessSTAC('https://cmr.earthdata.nasa.gov/stac/LPCLOUD')

    if verbose == True:
        print('Connected to NASA Earthdata (LPCLOUD).')

    # Define start and end
    start = str(year) + '-01-01'
    end = str(year) + '-12-31'

    # Cycle through all polygons
    for poly in area.index: # For each polygon...
        try: # Getting around server issues every so-often...         
            # Check if polygon has been completed
            #if obs_fill_S30[str(poly)].null_count() == len(obs_fill_S30): # All completed polygons should have at least 1 fill pixel recorded
            if area.loc[poly, progressTrack] != 1:
                print('--------------------')
                print('...Polygon ' + str(poly) + ' of ' + str(len(area)) + '...')

                # Convert polars dataframes to lazy mode for faster processing
                obs_land_L30 = obs_land_L30.lazy()
                obs_land_S30 = obs_land_S30.lazy()
                obs_water_L30 = obs_water_L30.lazy()
                obs_water_S30 = obs_water_S30.lazy()
                obs_snow_L30 = obs_snow_L30.lazy()
                obs_snow_S30 = obs_snow_S30.lazy()
                obs_aerosolL_L30 = obs_aerosolL_L30.lazy()
                obs_aerosolL_S30 = obs_aerosolL_S30.lazy()
                obs_aerosolW_L30 = obs_aerosolW_L30.lazy()
                obs_aerosolW_S30 = obs_aerosolW_S30.lazy()
                obs_aerosolS_L30 = obs_aerosolS_L30.lazy()
                obs_aerosolS_S30 = obs_aerosolS_S30.lazy()
                obs_adjacent_L30 = obs_adjacent_L30.lazy()
                obs_adjacent_S30 = obs_adjacent_S30.lazy()
                obs_shadow_L30 = obs_shadow_L30.lazy()
                obs_shadow_S30 = obs_shadow_S30.lazy()
                obs_cloud_L30 = obs_cloud_L30.lazy()
                obs_cloud_S30 = obs_cloud_S30.lazy()
                obs_fill_L30 = obs_fill_L30.lazy()
                obs_fill_S30 = obs_fill_S30.lazy()

                if metadata == True:
                    meta = meta.lazy()
                    st = time.time()
                    
                # Get bboxes based on area
                bboxLL = pputil.poly2bbox(area.loc[[poly]], 'lat/lon') # For STAC search
                bbox, epsg = pputil.poly2bbox(area.loc[[poly]], proj) # For stackstac

                if verbose == True:
                    print('Bounding boxes created and projection defined (EPSG:' + str(epsg) + ').')

                if metadata == True: # Record info in Metadata
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_bbox')
                                               .then(et - st) # Time (seconds) to create bboxes
                                               .otherwise(pl.col(str(poly)))
                                               .alias(str(poly))) 
                    st = time.time()
                    
                # Get all L30 items
                itemsL30 = catalog.search(bbox = bboxLL, datetime = f'{start}/{end}', collections = ['HLSL30_2.0'], limit = 100).item_collection() 

                if verbose == True: 
                    print('Nearby L30 images found on NASA EarthData (n = ' + str(len(itemsL30)) + ').')

                if metadata == True: # Record info in Metadata
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_items_L30')
                                               .then(et - st) # Time (seconds) to find L30 items on STAC
                                               .otherwise(pl.col(str(poly)))
                                               .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_items_L30')
                                               .then(len(itemsL30)) # Number of L30 items found on STAC
                                               .otherwise(pl.col(str(poly)))
                                               .alias(str(poly)))   
                    st = time.time()        

                # Get all S30 items
                itemsS30 = catalog.search(bbox = bboxLL, datetime = f'{start}/{end}', collections = ['HLSS30_2.0'], limit = 100).item_collection()   

                if verbose == True:
                    print('Nearby S30 images found on NASA EarthData (n = ' + str(len(itemsS30)) + ').')   

                if metadata == True: # Record info in Metadata
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_items_S30')
                                               .then(et - st) # Time (seconds) to find S30 items on STAC
                                               .otherwise(pl.col(str(poly)))
                                               .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_items_S30')
                                               .then(len(itemsS30)) # Number of S30 items found on STAC
                                               .otherwise(pl.col(str(poly)))
                                               .alias(str(poly)))   
                    st = time.time()

                # Create reprojected stackstac xarray for L30
                if len(itemsL30) > 0:
                    with warn.catch_warnings():
                        warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac
                        l30_stack = ss.stack(itemsL30, 
                                            assets = ['Fmask'], 
                                            epsg = epsg, 
                                            resolution = 30, 
                                            bounds = bbox, 
                                            resampling = Resampling.nearest, # discrete
                                            chunksize = (1, -1, -1, -1), # 1 time-step, all bands, full extent
                                            xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                                            dtype = 'uint8', # Reduces size to 12.5% of default (float64)
                                            fill_value = np.uint8(255), # 255 # Specifying np.dtype required for ss 0.5.1
                                            rescale =  False,
                                            gdal_env = gdalEnv)
                    
                    if verbose == True:
                        print('Created reprojected stackstac L30 cube (n = ' + str(l30_stack.shape[0]) + ').') 

                if len(itemsL30) == 0: 
                    l30_stack = xr.DataArray([]) # Create empty array             

                if metadata == True: # Record info in Metadata
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_stack_L30')
                                            .then(l30_stack.shape[0]) # Number of time-steps in initial (reprojected) stackstac array for L30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))             

                # Remove bad time-steps (NaT) from cube based on scene-level metadata
                if l30_stack.shape[0] > 0:
                    l30_remove = pputil.removeBadScenes(l30_stack, 100, verbose) # No cloud cover filtering

                if l30_stack.shape[0] == 0:
                    l30_remove = xr.DataArray([]) # Create empty array

                if metadata == True:
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_remove_L30')
                                            .then(l30_remove.shape[0]) # Number of time-steps after removing NaT dates for L30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))

                # Merge images from observations on the same day
                if l30_remove.shape[0] > 0:
                    l30_merge = pputil.sameDayMerge(l30_remove, 'mask', 'L30', '', 'hls', verbose)

                if l30_remove.shape[0] == 0:
                    l30_merge = xr.DataArray([]) # Create empty array 

                if metadata == True: # Record info in Metadata
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_lazy_L30')
                                            .then(et - st) # Time (seconds) to do all the lazy xarray piping for L30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30')
                                            .then(l30_merge.shape[0]) # Number of time-steps after same-day merge for L30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'gb_merge_L30')
                                            .then(l30_merge.nbytes * 1e-9) # Size (GB) for merged L30 cube
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    st = time.time()               

                # Create reprojected stackstac xarray for S30
                if len(itemsS30) > 0:
                    with warn.catch_warnings():
                        warn.simplefilter('ignore') # Avoids infer date time depreciation warnings from stackstac
                        s30_stack = ss.stack(itemsS30, 
                                            assets = ['Fmask'], 
                                            epsg = epsg, 
                                            resolution = 30, 
                                            bounds = bbox, 
                                            resampling = Resampling.nearest, # discrete
                                            chunksize = (1, -1, -1, -1), # 1 time-step, all bands, full extent
                                            xy_coords = 'center', # Coordinates are for each pixel's centroid (xarray convention)
                                            dtype = 'uint8', # Reduces size to 12.5% of default (float64)
                                            fill_value = np.uint8(255), # 255 # Specifying np.dtype required for ss 0.5.1
                                            rescale =  False,
                                            gdal_env = gdalEnv)
                        
                    if verbose == True:
                        print('Created reprojected stackstac S30 cube (n = ' + str(s30_stack.shape[0]) + ').')

                if len(itemsS30) == 0: 
                    s30_stack = xr.DataArray([]) # Create empty array                            

                if metadata == True: # Record info in Metadata
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_stack_S30')
                                            .then(s30_stack.shape[0]) # Number of time-steps in initial (reprojected) stackstac array for S30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))

                # Remove bad time-steps from cube based on scene-level metadata
                if s30_stack.shape[0] > 0:
                    s30_remove = pputil.removeBadScenes(s30_stack, 100, verbose) # No cloud cover filtering

                if s30_stack.shape[0] == 0:
                    s30_remove = xr.DataArray([]) # Create empty array                    

                if metadata == True: # Record info in Metadata
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_remove_S30')
                                            .then(s30_remove.shape[0]) # Number of time-steps after removing NaT dates for S30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))

                # Merge images from observations on the same day
                if s30_remove.shape[0] > 0:
                    s30_merge = pputil.sameDayMerge(s30_remove, 'mask', 'S30', '', 'hls', verbose) 

                if s30_remove.shape[0] == 0:
                    s30_merge = xr.DataArray([]) # Create empty array                 

                if metadata == True: # Record info in Metadata
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_lazy_S30')
                                            .then(et - st) # Time (seconds) to do all the lazy xarray piping for S30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30')
                                            .then(s30_merge.shape[0]) # Number of time-steps after same-day merge for S30
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'gb_merge_S30')
                                            .then(s30_merge.nbytes * 1e-9) # Size (GB) for merged S30 cube
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))   

                    # Test alternative cloud cover filtering
                    st = time.time()
                    # Start with S30/L30 remove, don't have to do full (can just look at len(groupby 'time')... faster)
                    if l30_remove.shape[0] > 0: 
                        if l30_remove['eo:cloud_cover'].size == 1: # When only 1 image (rare), need different structure for time = ...
                            # Need np.tile because sometimes all cc values are equal, giving 1 array size that is incorrect
                            l30_cc_vals = np.tile(np.array([l30_remove['eo:cloud_cover'].values]), l30_remove.shape[0])
                        if l30_remove['eo:cloud_cover'].size > 1:    
                            l30_cc_vals = l30_remove['eo:cloud_cover'].values

                        l30_90 = l30_remove.sel(time = l30_cc_vals <= 90)
                        l30_90 = l30_90['time'] = l30_90['time'].dt.floor('1D')
                        l30_75 = l30_remove.sel(time = l30_cc_vals <= 75)
                        l30_75 = l30_75['time'] = l30_75['time'].dt.floor('1D')
                        l30_50 = l30_remove.sel(time = l30_cc_vals <= 50)
                        l30_50 = l30_50['time'] = l30_50['time'].dt.floor('1D')
                        l30_10 = l30_remove.sel(time = l30_cc_vals <= 10)
                        l30_10 = l30_10['time'] = l30_10['time'].dt.floor('1D')

                    if l30_remove.shape[0] == 0:
                        l30_90 = xr.DataArray([]) # Create empty array
                        l30_75 = xr.DataArray([]) # Create empty array
                        l30_50 = xr.DataArray([]) # Create empty array
                        l30_10 = xr.DataArray([]) # Create empty array 

                    # Record info in Metadata
                    if len(l30_90) > 0: # Rarely, this one is empty (no < 90% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_90')
                                                .then(len(l30_90.groupby('time'))) # Number of time-steps in merged L30 cube (90+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(l30_90) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_90').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))

                    if len(l30_75) > 0: # Rarely, this one is empty (no < 75% cloud images)                      
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_75')
                                                .then(len(l30_75.groupby('time'))) # Number of time-steps in merged L30 cube (75+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(l30_75) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_75').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))

                    if len(l30_50) > 0: # Rarely, this one is empty (no < 50% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_50')
                                                .then(len(l30_50.groupby('time'))) # Number of time-steps in merged L30 cube (50+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(l30_50) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_50').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))

                    if len(l30_10) > 0: # Rarely, this one is empty (no < 10% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_10')
                                                .then(len(l30_10.groupby('time'))) # Number of time-steps in merged L30 cube (10+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(l30_10) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_L30_10').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))  
                            
                    if s30_remove.shape[0] > 0:
                        if s30_remove['eo:cloud_cover'].size == 1: # When only 1 image (rare), need different structure for time = ...
                            # Need np.tile because sometimes all cc values are equal, giving 1 array size that is incorrect
                            s30_cc_vals = np.tile(np.array([s30_remove['eo:cloud_cover'].values]), s30_remove.shape[0])
                        if s30_remove['eo:cloud_cover'].size > 1:      
                            s30_cc_vals = s30_remove['eo:cloud_cover'].values

                        s30_90 = s30_remove.sel(time = s30_cc_vals <= 90)
                        s30_90 = s30_90['time'] = s30_90['time'].dt.floor('1D')
                        s30_75 = s30_remove.sel(time = s30_cc_vals <= 75)
                        s30_75 = s30_75['time'] = s30_75['time'].dt.floor('1D')
                        s30_50 = s30_remove.sel(time = s30_cc_vals <= 50)
                        s30_50 = s30_50['time'] = s30_50['time'].dt.floor('1D')
                        s30_10 = s30_remove.sel(time = s30_cc_vals <= 10)
                        s30_10 = s30_10['time'] = s30_10['time'].dt.floor('1D')

                    if s30_remove.shape[0] == 0:
                        s30_90 = xr.DataArray([]) # Create empty array
                        s30_75 = xr.DataArray([]) # Create empty array
                        s30_50 = xr.DataArray([]) # Create empty array
                        s30_10 = xr.DataArray([]) # Create empty array 

                    # Record info in Metadata       
                    if len(s30_90) > 0: # Rarely, this one is empty (no < 90% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_90')
                                                .then(len(s30_90.groupby('time'))) # Number of time-steps in merged S30 cube (90+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(s30_90) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_90').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))

                    if len(s30_75) > 0: # Rarely, this one is empty (no < 75% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_75')
                                                .then(len(s30_75.groupby('time'))) # Number of time-steps in merged S30 cube (75+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(s30_75) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_75').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))

                    if len(s30_50) > 0: # Rarely, this one is empty (no < 50% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_50')
                                                .then(len(s30_50.groupby('time'))) # Number of time-steps in merged S30 cube (50+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(s30_50) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_50').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))

                    if len(s30_10) > 0: # Rarely, this one is empty (no < 10% cloud images)
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_10')
                                                .then(len(s30_10.groupby('time'))) # Number of time-steps in merged S30 cube (10+% cloud filter)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                    if len(s30_10) == 0:
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_merge_S30_10').then(0).otherwise(pl.col(str(poly))).alias(str(poly)))                                                

                    if (verbose == True) & ((l30_remove.shape[0] > 0) | (s30_remove.shape[0] > 0)):
                        print('Tested impact of cloud cover filtering (90, 75, 50, 10).')

                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_alternative')
                                             .then(et - st) # Time (seconds) to gather the alternative cloud cover information
                                             .otherwise(pl.col(str(poly)))
                                             .alias(str(poly)))
                    st = time.time()

                # Concat and Load
                if (l30_merge.shape[0] > 0) & (s30_merge.shape[0] > 0):    
                    fmask = xr.concat((l30_merge, s30_merge), dim = 'time').sortby('time')
                    fmask = pputil.loadXR(fmask) 
                    if verbose == True:
                        print('Combined into fmask cube and loaded into memory (n = ' + str(fmask.shape[0]) + ').')
                if (l30_merge.shape[0] > 0) & (s30_merge.shape[0] == 0):  
                    fmask = pputil.loadXR(l30_merge)
                    if verbose == True:
                        print('Loaded into memory (n = ' + str(fmask.shape[0]) + ').')
                if (s30_merge.shape[0] > 0) & (l30_merge.shape[0] == 0):  
                    fmask = pputil.loadXR(s30_merge)                
                    if verbose == True:
                        print('Loaded into memory (n = ' + str(fmask.shape[0]) + ').')
                if (l30_merge.shape[0] == 0) & (s30_merge.shape[0] == 0):
                    fmask = xr.DataArray([], dims = 'time') # Create empty array (with time dim to let metadata fill)

                if metadata == True:
                    et = time.time()

                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_load')
                                            .then(fmask.shape[0]) # Number of time-steps for combined cube loaded into memory
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'nunique_load')
                                            .then(len(np.unique(fmask.time))) # Number of unique time-steps in combined cube
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_obs_load')
                                            #.then(fmask.shape[0] * fmask.shape[-2] * fmask.shape[-1]) 
                                            .then(fmask.size)
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'gb_load')
                                            .then(fmask.nbytes * 1e-9) # Size (GB) for combined cube
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_load')
                                            .then(et - st) # Time (seconds) to load combined cube into memory 
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))
                    st = time.time()                     

                # Convert into Fmask categories
                if fmask.shape[0] > 0: 
                    # fill (10) > cloud (9) > shadow (8) > cloud adjacent (7) > aerosol (snow: 6, water: 5, land: 4) > snow (3) > water (2) > land (1)
                    fmask = pputil.convertFmask(fmask, format = 'categories')

                    if verbose == True:
                        print('Converted fmask cube into 10 categories.')

                if metadata == True:
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_convert')
                                            .then(et - st) # Time (seconds) to convert bit-packed Fmask values to 0-10 categories
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))

                # Border cleaning
                if borderCleaning != '':
                    if metadata == True:
                        st = time.time()
                        
                    if fmask.shape[0] > 0:
                        if area.loc[poly, borderCleaning] < 100: # Only mask for border squares
                            with warn.catch_warnings():
                                warn.simplefilter('ignore') # Avoids invalid value encountered in cast 
                                fmask = fmask.rio.clip(roi.geometry.values, drop = False) # Sets outside border to 0 for uint8

                                if verbose == True:
                                    print('Observations outside ROI (' + str(round(100 - area.loc[poly, borderCleaning], 3)) + '%) masked (value 0).')
                        
                    if metadata == True:
                        et = time.time()
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_mask')
                                                .then(et - st) # Time (seconds) to mask with Canada border (should be ~0 for non-border squares)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 'percentage')
                                                .then(area.loc[poly, borderCleaning]) # Percent of tile that falls within Canada border, retrieved from area gdf
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))

                # Get pixel counts by category for each time-step in dataframes
                if metadata == True:
                    st = time.time()

                if fmask.shape[0] > 0:
                    for timestep in range(len(fmask.time)): # For each time-step
                        df = pl.DataFrame(data = {'cat': list(range(1,11)), 'count': [0] * 10}) # Blank dataframe with 0s for each category

                        unique, counts = cp.unique(fmask[timestep].values, return_counts = True) # Faster than np.unique (same syntax)

                        unique = unique.get() # cupy to numpy for working with values
                        counts = counts.get()

                        for cat in unique: # For category existing in timestep
                            if cat > 0: # Account for mask being applied...
                                df = df.with_columns(pl.when(pl.col('cat') == cat)
                                                    .then(counts[int(np.where(unique == cat)[0][0])])
                                                    .otherwise(pl.col('count'))
                                                    .alias('count')) # Add category pixel count to df

                        day = str(fmask.time[timestep].values)[0:10] # Get day of timestep
                        day = dt.datetime.strptime(day, '%Y-%m-%d').date()

                        if fmask[timestep].constellation == 'L30': # For L30, add the count to correct spot in df
                            obs_land_L30 = obs_land_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                        .then(df[0, 'count'])
                                                                        .otherwise(pl.col(str(poly)))
                                                                        .alias(str(poly)))
                            obs_water_L30 = obs_water_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                            .then(df[1, 'count'])
                                                                            .otherwise(pl.col(str(poly)))
                                                                            .alias(str(poly)))
                            obs_snow_L30 = obs_snow_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                        .then(df[2, 'count'])
                                                                        .otherwise(pl.col(str(poly)))
                                                                        .alias(str(poly)))
                            obs_aerosolL_L30 = obs_aerosolL_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[3, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_aerosolW_L30 = obs_aerosolW_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[4, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_aerosolS_L30 = obs_aerosolS_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[5, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_adjacent_L30 = obs_adjacent_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[6, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_shadow_L30 = obs_shadow_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                            .then(df[7, 'count'])
                                                                            .otherwise(pl.col(str(poly)))
                                                                            .alias(str(poly)))
                            obs_cloud_L30 = obs_cloud_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                            .then(df[8, 'count'])
                                                                            .otherwise(pl.col(str(poly)))
                                                                            .alias(str(poly)))
                            obs_fill_L30 = obs_fill_L30.with_columns(pl.when(pl.col('Date') == day)
                                                                    .then(df[9, 'count'])
                                                                    .otherwise(pl.col(str(poly)))
                                                                    .alias(str(poly)))

                        if fmask[timestep].constellation == 'S30': # Same for S30...
                            obs_land_S30 = obs_land_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                        .then(df[0, 'count'])
                                                                        .otherwise(pl.col(str(poly)))
                                                                        .alias(str(poly)))
                            obs_water_S30 = obs_water_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                            .then(df[1, 'count'])
                                                                            .otherwise(pl.col(str(poly)))
                                                                            .alias(str(poly)))
                            obs_snow_S30 = obs_snow_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                        .then(df[2, 'count'])
                                                                        .otherwise(pl.col(str(poly)))
                                                                        .alias(str(poly)))
                            obs_aerosolL_S30 = obs_aerosolL_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[3, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_aerosolW_S30 = obs_aerosolW_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[4, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_aerosolS_S30 = obs_aerosolS_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[5, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_adjacent_S30 = obs_adjacent_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                                .then(df[6, 'count'])
                                                                                .otherwise(pl.col(str(poly)))
                                                                                .alias(str(poly)))
                            obs_shadow_S30 = obs_shadow_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                            .then(df[7, 'count'])
                                                                            .otherwise(pl.col(str(poly)))
                                                                            .alias(str(poly)))
                            obs_cloud_S30 = obs_cloud_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                            .then(df[8, 'count'])
                                                                            .otherwise(pl.col(str(poly)))
                                                                            .alias(str(poly)))
                            obs_fill_S30 = obs_fill_S30.with_columns(pl.when(pl.col('Date') == day)
                                                                    .then(df[9, 'count'])
                                                                    .otherwise(pl.col(str(poly)))
                                                                    .alias(str(poly)))
                            
                    if verbose == True:
                        print('Documented category counts for all time-steps.') 

                if fmask.shape[0] == 0:
                    df = pl.DataFrame(data = {'cat': list(range(1,11)), 'count': [0] * 10}) # Blank dataframe with 0s for each category                      

                if metadata == True:
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 'n_obs_ts')
                                            .then(df.sum()['count'][0]) # Number of non-masked observations in single time-step
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly)))                     
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_counts')
                                            .then(et - st) # Time (seconds) to put pixel counts in proper places in dataframes
                                            .otherwise(pl.col(str(poly)))
                                            .alias(str(poly))) 

                # For selected polygons, download Fmask cube as NetCDF
                if download != '':
                    if metadata == True:
                        st = time.time()

                    if fmask.shape[0] > 0:
                        if area.loc[poly, download] == 1: # Only download download polys
                            pputil.downloadNC(fmask, downloadPath, 'fmask_' + str(poly) + '_' + str(year) + '.nc', form = 'hls', encoding = 'set1')

                            if verbose == True:
                                print('Downloaded fmask cube as NetCDF file.')

                    if metadata == True:
                        et = time.time()
                        meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_download')
                                                .then(et - st)  # Time (seconds) to download cube (should be ~0 for non-download squares)
                                                .otherwise(pl.col(str(poly)))
                                                .alias(str(poly)))                                                        

                # Export CSVs (Observations, Metadata)
                if metadata == True:
                    st = time.time()

                # Collect lazy dataframes 
                obs_dfs = [obs_land_L30, obs_land_S30,  obs_water_L30, obs_water_S30, obs_snow_L30, obs_snow_S30, obs_aerosolL_L30, obs_aerosolL_S30, 
                           obs_aerosolW_L30, obs_aerosolW_S30, obs_aerosolS_L30, obs_aerosolS_S30, obs_adjacent_L30, obs_adjacent_S30, obs_shadow_L30, 
                           obs_shadow_S30, obs_cloud_L30, obs_cloud_S30, obs_fill_L30, obs_fill_S30]
                obs_dfs = pl.collect_all(obs_dfs) # Faster than collecting one by one

                # Unlist (easier to keep track of...)
                obs_land_L30 = obs_dfs[0]
                obs_land_S30 = obs_dfs[1]
                obs_water_L30 = obs_dfs[2]
                obs_water_S30 = obs_dfs[3]
                obs_snow_L30 = obs_dfs[4]
                obs_snow_S30 = obs_dfs[5]
                obs_aerosolL_L30 = obs_dfs[6]
                obs_aerosolL_S30 = obs_dfs[7]
                obs_aerosolW_L30 = obs_dfs[8]
                obs_aerosolW_S30 = obs_dfs[9]
                obs_aerosolS_L30 = obs_dfs[10]
                obs_aerosolS_S30 = obs_dfs[11]
                obs_adjacent_L30 = obs_dfs[12]
                obs_adjacent_S30 = obs_dfs[13]
                obs_shadow_L30 = obs_dfs[14]
                obs_shadow_S30 = obs_dfs[15]
                obs_cloud_L30 = obs_dfs[16]
                obs_cloud_S30 = obs_dfs[17]
                obs_fill_L30 = obs_dfs[18]
                obs_fill_S30 = obs_dfs[19] 

                obs_land_L30.write_csv(csv + '/Observations/obs_land_L30_' + str(year) + '.csv')
                obs_land_S30.write_csv(csv + '/Observations/obs_land_S30_' + str(year) + '.csv')
                obs_water_L30.write_csv(csv + '/Observations/obs_water_L30_' + str(year) + '.csv')
                obs_water_S30.write_csv(csv + '/Observations/obs_water_S30_' + str(year) + '.csv')
                obs_snow_L30.write_csv(csv + '/Observations/obs_snow_L30_' + str(year) + '.csv')
                obs_snow_S30.write_csv(csv + '/Observations/obs_snow_S30_' + str(year) + '.csv')
                obs_aerosolL_L30.write_csv(csv + '/Observations/obs_aerosolL_L30_' + str(year) + '.csv')
                obs_aerosolL_S30.write_csv(csv + '/Observations/obs_aerosolL_S30_' + str(year) + '.csv')
                obs_aerosolW_L30.write_csv(csv + '/Observations/obs_aerosolW_L30_' + str(year) + '.csv')
                obs_aerosolW_S30.write_csv(csv + '/Observations/obs_aerosolW_S30_' + str(year) + '.csv')
                obs_aerosolS_L30.write_csv(csv + '/Observations/obs_aerosolS_L30_' + str(year) + '.csv')
                obs_aerosolS_S30.write_csv(csv + '/Observations/obs_aerosolS_S30_' + str(year) + '.csv')
                obs_adjacent_L30.write_csv(csv + '/Observations/obs_adjacent_L30_' + str(year) + '.csv')
                obs_adjacent_S30.write_csv(csv + '/Observations/obs_adjacent_S30_' + str(year) + '.csv')
                obs_shadow_L30.write_csv(csv + '/Observations/obs_shadow_L30_' + str(year) + '.csv')
                obs_shadow_S30.write_csv(csv + '/Observations/obs_shadow_S30_' + str(year) + '.csv')
                obs_cloud_L30.write_csv(csv + '/Observations/obs_cloud_L30_' + str(year) + '.csv')
                obs_cloud_S30.write_csv(csv + '/Observations/obs_cloud_S30_' + str(year) + '.csv')
                obs_fill_L30.write_csv(csv + '/Observations/obs_fill_L30_' + str(year) + '.csv')
                obs_fill_S30.write_csv(csv + '/Observations/obs_fill_S30_' + str(year) + '.csv')       

                if metadata == True:
                    et = time.time()
                    meta = meta.with_columns(pl.when(pl.col('Metadata') == 't_csv')
                                               .then(et - st) # Time (seconds) to save csvs
                                               .otherwise(pl.col(str(poly)))
                                               .alias(str(poly)))
                    meta = meta.collect()
                    meta.write_csv(csv + '/Metadata/meta_' + str(year) + '.csv')
                
                print('Results saved to CSV.')
                        
                area.loc[poly, progressTrack] = 1 # Tracking progress in area column
                area.to_file(area_shp, engine = 'pyogrio')

                print('--------------------')

        except Exception as error: # Server errors every so often...
            #print('Skipping polygon', poly, '(', type(error).__name__, ').') # Brief error type reporting (API or CPLE for server related)
            print(traceback.format_exc()) # Gives more detail about error (for debugging)
            continue  

######################################################################################################################################################

#def sampleSpectralHLS() # For random60

######################################################################################################################################################
# Back End ###########################################################################################################################################
######################################################################################################################################################

# Access STAC catalog via link.
def accessSTAC(link):

    """
    Parameters:
    link (str): https link to STAC. 

    Returns:
    gdal environment settings and opened catalog. 
    
    """

    # Configure GDAL to access COGs (may only be required for EarthData, but just in case...)
    gdalEnv = ss.DEFAULT_GDAL_ENV.updated(dict(
        GDAL_DISABLE_READDIR_ON_OPEN = 'TRUE',
        GDAL_HTTP_COOKIEFILE = os.path.expanduser('~/cookies.txt'),
        GDAL_HTTP_COOKIEJAR = os.path.expanduser('~/cookies.txt'),
        GDAL_HTTP_MAX_RETRY = 10,
        GDAL_HTTP_RETRY_DELAY = 15, 
        GDAL_HTTP_UNSAFESSL = 'YES')) # Will pass to Dask cluster workers
    
    # Open the catalog
    catalog = pc.Client.open(link)

    return gdalEnv, catalog

######################################################################################################################################################

# Create observation and metadata csvs to fill with observationAvailabilityHLS().
def OAHLScsvs(path, polygons, years, metadata):

    """
    Parameters:
    path: path to store created csvs ('C:/path/to/csvs'). Will create 'Observations' and 'Metadata' folders. 

    polygons: GeoDataFrame or path to shp with index representing polygons.   

    year (list of int): years to create CSVs for. 

    metadata (bool): Whether (true) or not (false) to create accompanying metadata csv.
 
    Returns:
    CSVs output to path. 
    """

    # Building different csv
    cats = ['land', 'water', 'snow', 'aerosolL', 'aerosolW', 'aerosolS', 'adjacent', 'shadow', 'cloud', 'fill'] # From fmask
    sens = ['L30', 'S30']

    # CSV schema
    if metadata == False:
        obs_schema = OAHLSschema(polygons, metadata)
    if metadata == True:
        obs_schema, meta_schema = OAHLSschema(polygons, metadata)

    # Observations
    for year in years:

        # Get days based on year
        days = [dt for dt in rrule(DAILY, 
                                   dtstart = dt.datetime.strptime(str(year) + '-01-01', '%Y-%m-%d'), 
                                   until = dt.datetime.strptime(str(year) + '-12-31', '%Y-%m-%d'))]
        days = [day.date() for day in days] # Just date

        for cat in cats:

            for sen in sens:

                df = pl.DataFrame(schema = obs_schema)
                df = pl.concat([df, pl.Series('Date', days).to_frame()], how = 'diagonal')
                
                csv  = 'obs_' + cat + '_' + sen + '_' + str(year) + '.csv' 
                df.write_csv(path + '/Observations/' + csv) 

    # Metadata
    if metadata == True:

        # Metadata column values
        meta = ['t_bbox', # Time (seconds) to create bboxes
                'n_items_L30', # Number of L30 items found on STAC
                't_items_L30', # Time (seconds) to find L30 items on STAC
                'n_items_S30', # Number of S30 items found on STAC
                't_items_S30', # Time (seconds) to find S30 items on STAC
                #'n_obs_x', # Number of observations (pixels) along one spatial dimension
                #'n_obs_y', # Number of observations (pixels) along other spatial dimension
                'n_stack_L30', # Number of time-steps in initial (reprojected) stackstac array for L30
                'n_remove_L30', # Number of time-steps after removing NaT dates for L30
                'n_merge_L30', # Number of time-steps after same-day merge for L30
                'gb_merge_L30', # Size (GB) for merged L30 cube
                't_lazy_L30', # Time (seconds) to do all the lazy xarray piping for L30
                'n_stack_S30',  # Number of time-steps in initial (reprojected) stackstac array for S30
                'n_remove_S30', # Number of time-steps after removing NaT dates for S30
                'n_merge_S30', # Number of time-steps after same-day merge for S30
                'gb_merge_S30', # Size (GB) for merged S30 cube
                't_lazy_S30', # Time (seconds) to do all the lazy xarray piping for S30
                'n_merge_L30_90', # Number of time-steps in merged L30 cube, filtering out 90+% cloud cover scenes
                'n_merge_L30_75', # Number of time-steps in merged L30 cube, filtering out 75+% cloud cover scenes
                'n_merge_L30_50', # Number of time-steps in merged L30 cube, filtering out 50+% cloud cover scenes
                'n_merge_L30_10', # Number of time-steps in merged L30 cube, filtering out 10+% cloud cover scenes
                'n_merge_S30_90', # Number of time-steps in merged S30 cube, filtering out 90+% cloud cover scenes
                'n_merge_S30_75', # Number of time-steps in merged S30 cube, filtering out 75+% cloud cover scenes
                'n_merge_S30_50', # Number of time-steps in merged S30 cube, filtering out 50+% cloud cover scenes
                'n_merge_S30_10', # Number of time-steps in merged S30 cube, filtering out 10+% cloud cover scenes
                't_alternative', # Time (seconds) to gather the alternative cloud cover information
                'n_load', # Number of time-steps for combined cube loaded into memory (should equal n_merge_L30 + n_merge_S30)
                'nunique_load', # Number of unique time-steps (i.e., removing S30 and L30 on same day) in combined cube
                'n_obs_load', # Number of total observations in combined cube (should equal n_load * n_obs_x * n_obs_y)
                'gb_load', # Size (GB) for combined cube (should equal gb_merge_L30 + gb_merge_S30)
                't_load', # Time (seconds) to load combined cube into memory (should be the bulk of time required)
                't_convert', # Time (seconds) to convert bit-packed Fmask values to 0-10 categories
                'percentage', # Percent of tile that falls within Canada border, retrieved from geodataframe
                #'n_obs_na', # Number of "NA" values outside categories (e.g., < 1, > 10) - should be 0 unless masking applied at border (takes too long)
                't_mask', # Time (seconds) to mask with Canada border (if applicable) and count NAs
                'n_obs_ts', # Number of non-masked observations in single time-step (should equal n_obs_x * n_obs_y for non-masked polygons)
                't_counts', # Time (seconds) to put pixel counts in proper places in dataframes
                't_download', # Time (seconds) to download cube - should be 0 for all but select polygons
                't_csv'] # Time (seconds) to save csvs

        for year in years: 

            df = pl.DataFrame(schema = meta_schema)
            df = pl.concat([df, pl.Series('Metadata', meta).to_frame()], how = 'diagonal')

            csv  = 'meta_' + str(year) + '.csv' 
            df.write_csv(path + '/Metadata/' + csv)

######################################################################################################################################################

# Create schema for observationAvailabilityHLS csvs.
def OAHLSschema(polygons, metadata):

    """
    Parameters:
    polygons: GeoDataFrame or path to shp with index representing polygons.  

    metadata (bool): Whether (true) or not (false) to create accompanying metadata csv.

    Returns:
    Schema dictionaries for polars dataframes/CSVs
    """
    
    # Polygons
    if type(polygons) == str: # Path to shp given
        polygons = gpd.read_file(polygons, engine = 'pyogrio')

    poly_names = list(map(str, polygons.index.tolist()))

    # Observations schema
    poly_dtype = [pl.Float32] * len(polygons)
    poly_schema = {poly_names[col]: poly_dtype[col] for col in range(len(poly_names))}
    obs_schema = {'Date': pl.Date} # Dates
    obs_schema.update(poly_schema) # Dates as first in schema

    # Metadata schema
    if metadata == True:
        poly_dtype = [pl.Float64] * len(polygons)
        poly_schema = {poly_names[col]: poly_dtype[col] for col in range(len(poly_names))}
        meta_schema = {'Metadata': pl.String} # Metadata column
        meta_schema.update(poly_schema) # Metadata as first in schema

    if metadata == False:
        return obs_schema
    if metadata == True:
        return obs_schema, meta_schema

######################################################################################################################################################