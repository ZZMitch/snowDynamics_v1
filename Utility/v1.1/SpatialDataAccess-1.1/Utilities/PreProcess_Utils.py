######################################################################################################################################################
#
#   name:       PreProcess_Utils.py
#   contains:   Functions for pre-processing large datasets into desired format (e.g., HLS into clean xarray time-series)
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Built in
import glob
import os
import warnings as warn
import time
import re

# Open source
import geopandas as gpd
import numpy as np # import cupy as cp
import pandas as pd
import pyproj as pp
import rasterio as rio
import rioxarray as rxr
import xarray as xr
import stackstac as ss

from dask import diagnostics as ddiag
from rasterio import features as riof
from tqdm.notebook import tqdm
from shapely import geometry
#from retry import retry

# Recommended to have installed...
# https://unidata.github.io/netcdf4-python/ 
# https://flox.readthedocs.io/en/latest/
# https://pyogrio.readthedocs.io/en/latest/ 

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Need to fix xarray2GTs
# Could adjust cleanCube to specify unclear value (rather than requiring 1/0 mask with 0 being unclear)

######################################################################################################################################################
# Front-End ##########################################################################################################################################
######################################################################################################################################################

# Download all time-steps in xarray as GeoTiffs in set file system (download observations by year to limit impact of server/memory issues).
def xarray2GTs(cube, path, ID, rescale, fill, NAthresh): # This requires a re-write after chaning rescale and pixelMask/cleanCube...

    """
    Parameters:
    cube (in-memory xarray): Cube of interest for saving

    path (string): Folder to save GeoTiffs ('C:/path/to/file')

    ID (string): ID to define file system 

    rescale (float): value to rescale cube by (e.g., 0.0001 will change 9000 to 0.9). 1 = no rescaling. 

    fill (int): fill value to convert to NA. ''None'' = no fill to convert. 

    NAthresh (int): Highest cube time-step level NA % to include

    Results:
    GeoTiffs for each time-step downloaded to local file system (split into yearly folders)
    """

    # Identify start and end year 
    start = int(str(cube.time[0].values)[0:4])
    end = int(str(cube.time[-1].values)[0:4])

    for year in range(start, end + 1): # For each year from start of time-series to end...

        print(year)

        # Create folder
        fullPath = os.path.join(path, ID, str(year))
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)

        # Subset cube
        cube_yr = cube.where(cube['time.year'] == year, drop = True)

        if len(glob.glob1(fullPath, '*.tif')) < 2: # At less than 2 tifs in folder...

            # Load cube into memory
            print('Loading into memory...')   
            cube_yr = loadXR(cube_yr)

            # Reduce size and rescale
            cube_yr = rescaleFill(cube_yr, rescale, fill, verbose = True)

            # Download
            print('Downloading...')
            downloadGT(cube_yr, fullPath, ID)

        else:
            print('Tifs already downloaded.') 
######################################################################################################################################################

# Clean cube by removing time-steps with high NA % OR high unclear % (from mask) and/or pixel masking based on mask.
def cleanCube(cube, mask = [], timestep = '',  thresh = 100, rescale = 1, fill = '', pixel = False, reverseMask = False, verbose = True): 

    """
    Parameters:
    cube (xarray): cube to mask

    mask (xarray): mask to apply. Should be in form 1/True = do not mask (clear), 0/False = mask (unclear).
    - mask should have same time-steps as cube
    - not required if just doing timestepClean = 'NA'

    timestep (str): Type of time-step cleaning. '' = skip. 'NA': clean based on NA values. 'unclear': clean based on unclear values in mask

    thresh (int): Highest NA or unclear % time-step to include. 100 = no removing.

    pixel (bool): Whether (true) or not (false) to set cube pixels corresponding to unclear in mask to NA.

    reverseMask (bool): Whether (true) or not (false) to reverse the mask so only masked values remain (e.g., want to visualize cloudy pixels).
    - Only matters if pixelClean set to True

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube (xarray): cleaned cube
    """

    # Timestep-level cleaning
    if (timestepClean != '') & (thresh < 100):
        cube, mask = timestepClean(cube, mask, timestep, thresh, cleanMask = True, verbose = verbose)

    # Rescale
    if (rescale != 1) | (fill != ''):
        cube = rescaleFill(cube, rescale, fill, verbose)

    # Pixel-level cleaning
    if pixel == True:
        cube = pixelClean(cube, mask, reverseMask, verbose) # This breaks with longer dask-backed cubes (works fine once in memory)

    return cube

######################################################################################################################################################

######################################################################################################################################################
# Data Transfer ######################################################################################################################################
######################################################################################################################################################

# Compute xarray into memory.
#@retry()
def loadXR(cube):

    """
    Parameters: 
    cube (dask-basked xarray): Cube of interest

    Returns:
    cube (in-memory xarray)
    """

    with warn.catch_warnings():
        warn.simplefilter('ignore') # Avoids RuntimeWarning about All-NaN slices
        with ddiag.ProgressBar():
            with rio.Env(GDAL_HTTP_UNSAFESSL = 'YES') as env: # Avoids NRCan network blocking access - Also implemented in accessSTAC()
            # compute (returns python object fit in mem), persist (returns dask object), load (like compute but inplace)
                cube = cube.load() 
                
    return cube
######################################################################################################################################################

# Download xarray DataArray as netCDF file for later use.
def downloadNC(cube, path, file, form = '', encoding = 'set1'):

    """
    Parameters:
    cube (in-memory xarray): Cube of interest for saving

    path (string): Folder to save file ('C:/path/to/file')

    file (string): File to save ('file.nc')

    form (string): Type of data being saved, when special steps are required. Supports: 'hls'. '' = No special steps required. 

    encoding (str): Type of encodign to use.
    - 'set1' (default): zlib level 4 compression, with chunksizes set to 1 for time and full size for other dimensions. 

    Returns:
    NetCDF file saved in specified folder. 
    """

    if form == 'hls':
        # Adjust spec attribute to string for serialization into netCDF
        cube.attrs['spec'] = str(cube.attrs['spec'])  
    
    # Set encoding
    if encoding == 'set1': # Set 1: zlib level 4 compression, with chunksizes set to 1 for time and full size for other dimensions.
        if type(cube) == xr.core.dataarray.DataArray: # DataArray    
            encoding = {cube.name: {'zlib': True, 'complevel': 4, 'chunksizes': (1, 1, cube.shape[2], cube.shape[3])}}
        if type(cube) == xr.core.dataset.Dataset: # DataSet
            encoding = {var: {'zlib': True, 'complevel': 4, 'chunksizes': (1, cube.sizes['x'], cube.sizes['y'])} for var in cube.data_vars}
    else:
        encoding = None # No compression/other encoding
    
    cube.to_netcdf(path = os.path.join(path, file), 
                   mode = 'w', # Overwrite files (default)
                   format = 'NETCDF4', # HDF5 file with netCDF4 API (default)
                   engine = 'h5netcdf', # Default, recommended over netcdf4
                   encoding = encoding) 
######################################################################################################################################################
    
# Upload netCDF file as xarray DataArray or DataSet.
def uploadNC(nc, form = '', chunks = None):

    """
    Parameters:
    nc: saved NetCDF file

    form (str): Type of xarray to open. Supports 'dataarray', 'dataset'.
    - dataarray: Single variable NetCDF
    - dataset: Multi-variable NetCDF

    chunks: Chunks for dask-backed cube. None = Not dask-backed. {'time': 1, 'band': -1, 'x': -1, 'y': -1} is common (chunk by time-step only).

    Returns:
    cube: in-memory xarray
    """

    if form == 'dataarray': # Open as xarray DataArray
        cube = xr.open_dataarray(filename_or_obj = nc, 
                                engine = 'h5netcdf',  # Default, recommended over netcdf4
                                chunks = chunks, 
                                decode_coords = 'all', # Decode_coords get spatial_ref
                                mask_and_scale = False) # Mask will set _FillValue to NaN (changing dtype) and removing attr. Usually don't want this. 
        
    if form == 'dataset': # Open as xarray DataSet
        cube = xr.open_dataset(filename_or_obj = nc, 
                               engine = 'h5netcdf', # Default, recommended over netcdf4
                               chunks = chunks, 
                               decode_coords = 'all', # Decode_coords get spatial_ref,
                               mask_and_scale = False) # Mask will set _FillValue to NaN (changing dtype) and removing attr. Usually don't want this. 
    
    return cube
######################################################################################################################################################

# Download xarray DataArray as GeoTiffs for later use (currently supports HLS/Fmask).
def downloadGT(cube, path, ID):

    """
    Parameters: 
    cube (in-memory xarray): Cube of interest for saving

    path (string): Folder to save GeoTiffs ('C:/path/to/file')

    ID (string): Identifier to start file name for all saved GeoTiffs

    Results:
    GeoTiff saved in folder with name (FolderName_YYYY-MM-DD_Con (e.g., HLS_2023-08-15_L30.tif))
    """

    # Convert cube to dataset with bands as data variables
    if cube.ndim == 4: # Multiple bands (e.g., spectral cube)
        cube = cube.to_dataset(dim = 'band')
    else: # Single band (e.g., qa cube)
        cube = cube.to_dataset(name = 'Fmask')

    for i in tqdm(range(len(cube.time))): # For each timestep...
        timestep = cube.isel(time = i)
        date = str(timestep.time.values)[0:10]
        con = str(timestep.constellation.values)
        file = ID + '_' + date + '_' + con + '.tif'
        fullPath = os.path.join(path, file)

        if not os.path.exists(fullPath): # Don't overwrite
            timestep.rio.to_raster(fullPath)
######################################################################################################################################################
            
# Upload GeoTiff files as xarray (e.g., outputs from downloadGT()) based on provided form. 
def uploadGT(path, ID, form = '', names = []):

    """
    Parameters:
    path: Folder where GeoTiffs are saved ('C:/path/to/file')

    ID (string): Identifier to find GeoTiff files of interest

    form (string): Type of geotiff upload to complete. Supports: 'hls', 'snowDynamics'.
    - hls: HLS geotiffs to dataArray
    - snowDynamics: IMS or HLS snow dynamics to dataSet

    names (list): List of names to rename coordinates/variables. 
    - hls: Expects list of HLS band names
    - snowDynamics: Expects list of snowDynamics statistics (e.g., ['start', 'end', 'length', 'periods', 'status'])

    Results:
    cube (in-memory xarray)
    """

    if form == 'hls':

        # Get list of all tifs of interest
        tifs = glob.glob(os.path.join(path, ID + '_*.tif'))

        # Get date information from file names
        dates = [os.path.basename(i)[slice(*(-18,-8))] for i in tifs]
        dates = xr.Variable('time', pd.to_datetime(dates)) # Datetime > xarray variable

        # Get constellation information from file names
        cons = [os.path.basename(i)[slice(*(-7,-4))] for i in tifs]

        # Create DataArray
        cube = xr.concat([rxr.open_rasterio(tif) for tif in tifs], dim = dates)
        cube = cube.assign_coords(constellation = ('time', cons))

        cube['band'] = names

    if form == 'snowDynamics':

        # Get list of all tifs of interest
        tifs = glob.glob(os.path.join(path, '**', f'*{ID}*.tif'), recursive = True) # path, ID + '/**/*tif'

        # From first tif, get information to create dataset
        first = rxr.open_rasterio(tifs[0])
        cube = xr.Dataset(coords = dict(x = ('x', first['x'].values), y = ('y', first['y'].values))) # Blank dataset with just pixel grid
        cube.rio.write_crs(first.rio.crs, inplace = True) # Apply crs

        # Create winterYear variable
        num4 = [] # 4 digits (winterYears)
        for tif in tifs: # All tif filenames
            num4.append(re.findall(r'\d{4}', tif)[0]) # First set of 4 numbers in all filenames
        num4 = list(dict.fromkeys(num4)) # Ordered unique sets of 4 numbers

        winter = ['20' + num4[i][0:2] + '-20' + num4[i][2:4] for i in range(len(num4))] # Convert to proper convention
        winter = xr.Variable('winterYear', winter) # Set as variable

        # For each snow dynamic stat, create dataArray and add to dataSet
        for stat in names:
            cube['snow_' + stat] = xr.concat([rxr.open_rasterio(tif, chunks = 'auto', mask_and_scale = True) for tif in tifs if (stat in tif) & 
                                              (not stat + '_u' in tif)], dim = winter).squeeze() # Find correct tifs, with fix for HLS uncertainty 

    return cube
######################################################################################################################################################

######################################################################################################################################################
# Pixel Processing ###################################################################################################################################
###################################################################################################################################################### 

# Convert bitpacked Fmask band to more usable formats (e.g., for cloud-masking) using hierarchy approach.
def convertFmask(fmask, format, consideredClear = ['land', 'water']):
    
    """
    Parameters: 
    Fmask (xr): Fmask band xarray (bit-packed).

    format (str): Fmask output format.
    - 'clearUnclear': clear (True/1), unclear (False/0)
    - 'categories': fill (10) > cloud (9) > shadow (8) > cloud adjacent (7) > aerosol (snow: 6, water: 5, land: 4) > snow (3) > water (2) > land (1)
    Note: Hierarchical format (e.g., snow pixel only categorized if not already part of higer value class)

    consideredClear (list of str): categories to be considered clear. Only applies if format = 'clearUnclear'
    - Supports: 'land', 'water', 'snow', 'aerosol_land', 'aerosol_water', 'aerosol_snow', 'cloud_adjacent', 'shadow', 'cloud'

    Results:
    Unpacked Fmask xarray in requested format
    """

    # Bit values for each category (see convertFmask_Testing for process to find these). Contains unused bits for completeness (0 - 256).
    fill_bits = [255]
    cloud_bits = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63, 66, 67, 
                  70, 71, 74, 75, 78, 79, 82, 83, 86, 87, 90, 91, 94, 95, 98, 99, 102, 103, 106, 107, 110, 111, 114, 115, 118, 119, 122, 123, 126, 
                  127, 130, 131, 134, 135, 138, 139, 142, 143, 146, 147, 150, 151, 154, 155, 158, 159, 162, 163, 166, 167, 170, 171, 174, 175, 178, 
                  179, 182, 183, 186, 187, 190, 191, 194, 195, 198, 199, 202, 203, 206, 207, 210, 211, 214, 215, 218, 219, 222, 223, 226, 227, 230, 
                  231, 234, 235, 238, 239, 242, 243, 246, 247, 250, 251, 254]
    shadow_bits = [8, 9, 12, 13, 24, 25, 28, 29, 40, 41, 44, 45, 56, 57, 60, 61, 72, 73, 76, 77, 88, 89, 92, 93, 104, 105, 108, 109, 120, 121, 124, 
                   125, 136, 137, 140, 141, 152, 153, 156, 157, 168, 169, 172, 173, 184, 185, 188, 189, 200, 201, 204, 205, 216, 217, 220, 221, 232, 
                   233, 236, 237, 248, 249, 252, 253]
    adjacent_bits = [4, 5, 20, 21, 36, 37, 52, 53, 68, 69, 84, 85, 100, 101, 116, 117, 132, 133, 148, 149, 164, 165, 180, 181, 196, 197, 212, 213, 
                     228, 229, 244, 245]
    aerosolS_bits = [208, 209, 240, 241]
    aerosolW_bits = [224, 225]
    aerosolL_bits = [192, 193]
    snow_bits = [16, 17, 48, 49, 80, 81, 112, 113, 144, 145, 176, 177]
    water_bits = [32, 33, 96, 97, 160, 161]
    land_bits = [0, 1, 64, 65, 128, 129]

    # Clear/unclear format
    if format == 'clearUnclear':
        clear_bits = []

        # ID clear bits
        if 'land' in consideredClear:
            clear_bits.extend(land_bits)
        if 'water' in consideredClear:
            clear_bits.extend(water_bits)
        if 'snow' in consideredClear:
            clear_bits.extend(snow_bits)    
        if 'aerosol_land' in consideredClear:
            clear_bits.extend(aerosolL_bits)
        if 'aerosol_water' in consideredClear:
            clear_bits.extend(aerosolW_bits)  
        if 'aerosol_snow' in consideredClear:
            clear_bits.extend(aerosolS_bits)
        if 'cloud_adjacent' in consideredClear:
            clear_bits.extend(adjacent_bits)
        if 'shadow' in consideredClear:
            clear_bits.extend(shadow_bits)
        if 'cloud' in consideredClear:
            clear_bits.extend(cloud_bits)
        
        # Reclassify into clear/unclear
        fmask.values = fmask.isin(clear_bits) # Boolean array

    # Categories format
    if format == 'categories':

        # Mapped values as list of tuples
        fill_map = list(tuple(zip(fill_bits, [10] * len(fill_bits)))) # Fill bits = 10
        cloud_map = list(tuple(zip(cloud_bits, [9] * len(cloud_bits)))) # Cloud bits = 9
        shadow_map = list(tuple(zip(shadow_bits, [8] * len(shadow_bits)))) # Shadow bits = 8 
        adjacent_map = list(tuple(zip(adjacent_bits, [7] * len(adjacent_bits)))) # Cloud adjacent bits = 7
        aerosolS_map = list(tuple(zip(aerosolS_bits, [6] * len(aerosolS_bits)))) # Aerosol - snow bits = 6
        aerosolW_map = list(tuple(zip(aerosolW_bits, [5] * len(aerosolW_bits)))) # Aerosol - water bits = 5
        aerosolL_map = list(tuple(zip(aerosolL_bits, [4] * len(aerosolL_bits)))) # Aerosol - land bits = 4
        snow_map = list(tuple(zip(snow_bits, [3] * len(snow_bits)))) # Snow bits = 3
        water_map = list(tuple(zip(water_bits, [2] * len(water_bits)))) # Water bits = 2
        land_map = list(tuple(zip(land_bits, [1] * len(land_bits)))) # Land bits = 1

        # All mapped values together
        maps = []
        maps.extend(fill_map + cloud_map + shadow_map + adjacent_map + aerosolS_map + aerosolW_map + aerosolL_map + snow_map + water_map + land_map)

        idx, val = np.asarray(maps).T
        map_array = np.zeros(idx.max() + 1, dtype = 'uint8')
        map_array[idx] = val
        
        # Apply map to values
        fmask.values = map_array[fmask]

    return fmask
######################################################################################################################################################

# Rescale cube and set fill value to NA.
def rescaleFill(cube, rescale = 1, fill = '', verbose = True):

    """
    Parameters: 
    cube (in-memory xarray): xarray time-series cube (e.g., from stackstac or odc-stac)

    rescale (float): value to rescale cube by (e.g., 0.0001 will change 9000 to 0.9). 1 = no rescaling. 

    fill (int): fill value to convert to NA. '' = no fill to convert. 

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube (xarray): Reduced size time-series cube
    """

    # Rescale and set fill to NA
    if (rescale != 1) & (fill != ''): #  Rescale and fill
        cube = cube.where(cube != fill) * rescale

        if verbose == True:
            print(str(fill) + ' set to NA and other values rescaled by ' + str(rescale) + '.')
    
    if (rescale != 1) & (fill == ''): # Only  rescale
        cube = cube * rescale

        if verbose == True:
            print('Values rescaled by ' + str(rescale), '.')
    
    if (rescale == 1) & (fill != ''): # Only fill
        cube = cube.where(cube != fill)

        if verbose == True:
            print(fill, 'set to NA.')

    return cube 
######################################################################################################################################################

# Apply pixel level mask to cube.
def pixelClean(cube, mask = [], reverseMask = False, verbose = True): 

    """
    Parameters:
    cube (xarray): cube to mask

    mask (in-memory xarray): mask to apply. Should be in form 1/True = do not mask (clear), 0/False = mask (unclear).
    - mask should have same time-steps as cube

    reverseMask (bool): Whether (true) or not (false) to reverse the mask so only masked values remain (e.g., want to visualize cloudy pixels).

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube (xarray): cleaned cube
    """

    st = time.time()
    # If mask does not have same time-steps as cube, raise error
    if np.array_equal(cube.time.values, mask.time.values) == False:
        raise Exception('Time-step mismatch between cube and mask.')   

    # Match mask to cube dimensions (i.e., # bands)
    mask = np.broadcast_to(mask, shape = cube.shape)
    et = time.time()
    print(et - st)

    # Mask cube based on mask
    if reverseMask == False: 
        cube = xr.where(mask == 0, np.nan, cube)

        if verbose == True:
            print('Unclear pixels masked.')

    if reverseMask == True:
        cube = xr.where(mask == 1, np.nan, cube)

        if verbose == True:
            print('Clear pixels masked.')

    return cube

######################################################################################################################################################
# Spatial ############################################################################################################################################
######################################################################################################################################################

# Gets bounding box in specified projection based on input polygon.
def poly2bbox(poly, proj):

    """
    Parameters:
    poly (string or gdf): path to shp ('C:/path/to/poly.shp') or GeoDataFrame

    proj (string):  Projection of output bbox. 'UTM': local UTM zone. 'lat/lon': EPSG:4326. 'poly': same as input polygon. 

    Returns:
    bbox (tuple): Coordinates of 4 corners in specified projection
    """

    # Load poly
    if type(poly) == str: # Path to shp given
        aoi = gpd.read_file(poly, engine = 'pyogrio')
    else:
        aoi = poly # GeoDataFrame

    if proj == 'UTM':

        aoi = aoi.to_crs('EPSG:4326')

        # Get centroid lat/lon coordinates
        with warn.catch_warnings():
            warn.simplefilter('ignore')
            cent = aoi.centroid

        lon = float(cent.x.iloc[0])
        lat = float(cent.y.iloc[0])

        # Get UTM EPSG from lat/lon
        utm = pp.database.query_utm_crs_info(datum_name = 'WGS 84', 
                                            area_of_interest = pp.aoi.AreaOfInterest(lon, lat, lon, lat))
        epsg = int(utm[0].code)
        crs = pp.CRS.from_epsg(epsg)

        # Reproject poly and get bbox
        aoi = aoi.to_crs(crs)
        bbox = riof.bounds(aoi)

        return bbox, epsg

    if proj == 'lat/lon':

        aoi = aoi.to_crs('EPSG:4326')
        bbox = riof.bounds(aoi)

        return bbox # Don't care about epsg in this case
    
    if proj == 'poly':

        epsg = aoi.crs.to_epsg()
        bbox = riof.bounds(aoi)

        return bbox, epsg

######################################################################################################################################################

# Gets UTM and lat/lon bounding box based on input lat/lon and edge size (m).
def latlon2bboxes(lat, lon, edge):

    """
    Parameters:
    lat (float): Latitude in decimal degrees

    lon (float): Longitude in decimal degrees

    edge (float): Edge size of bounding box in meters

    Returns: 
    bounding box, projection epsg
    """

    utm = pp.database.query_utm_crs_info(datum_name = 'WGS 84', area_of_interest = pp.aoi.AreaOfInterest(lon, lat, lon, lat))

    epsg = int(utm[0].code)

    trans = pp.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy = True)
    transInv = pp.Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy = True)

    utmCoords = trans.transform(lon, lat)

    E = utmCoords[0] + edge
    W = utmCoords[0] - edge
    N = utmCoords[1] + edge
    S = utmCoords[1] - edge
    poly = [[W, S], [E, S], [E, N], [W, N], [W, S]]

    polyLL = [list(transInv.transform(x[0], x[1])) for x in poly]
    bboxLL = {'type': 'Polygon', 'coordinates': [polyLL]}

    bboxUTM = {'type': 'Polygon', 'coordinates': [poly]}
    bboxUTM = riof.bounds(bboxUTM)   

    return bboxUTM, bboxLL, epsg 

######################################################################################################################################################

# Create fishnet at specified size based on input polygon.
def fishnet(poly, proj, size, overlap = 0, intersect = '', centroid = True, output = 'gdf', verbose = True):

    """
    Parameters:
    poly (string): Path to shp ('C:/path/to/poly.shp') to create fishnet from. Note: Best if one polygon... will dissolve multiple (but slow),

    proj (int): Projection

    size (int): Size of fishnet squares. In same unit as projection. 

    overlap (int): Size of overlap between fishnet squares. In same unit as projection. 0 = no overlap. Will increase size of squares. 

    intersect (string): Handles filtering square by intersection with a GeoDataFrame. '' = do not consider intersections (keep all squares). 
    Otherwise, path to shp. If same path as poly, does not load twice. 

    centroid (boolean): Whether (true) or not (false) to output square centroids as points along with fishnet. Take output filename and adds '_pts'.

    output (string): 'gdf' = geoDataFrame. Else = filename of shapefile. 

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    fishnet (shp or gdf): Fishnet polygons
    """
    # Load polygon,  dissolve to remove internal boundaries
    gdf = gpd.read_file(poly, engine = 'pyogrio').to_crs(proj)

    if len(gdf) > 1:
        gdf = gdf.dissolve()

    # Drop all useless columns 
    gdf = gpd.GeoDataFrame(geometry = gdf['geometry'])

    if verbose == True:
        print('Loaded polygon.')

    # Get bounds information
    bounds = gdf.total_bounds
    minX, minY, maxX, maxY = bounds

    # Create fishnet
    x, y = (minX, minY)
    fishnet = []

    while y <= maxY:
        while x <= maxX:
            geom = geometry.Polygon([(x - overlap, y - overlap), 
                                     (x - overlap, y + size + overlap), 
                                     (x + size + overlap, y + size + overlap), 
                                     (x + size + overlap, y - overlap), 
                                     (x - overlap,  y - overlap)])
            fishnet.append(geom)
            x += size
        x = minX
        y += size

    fishnet = gpd.GeoDataFrame(fishnet, columns = ['geometry']).set_crs(proj)   

    if verbose == True:
        print('Created fishnet with', len(fishnet), 'squares.') 

    # Filter by intersection
    if intersect != '':

        if intersect == poly:
            gdfI = gdf

        else:
            gdfI = gpd.read_file(intersect, engine = 'pyogrio').to_crs(proj)

            if verbose == True:
                print('Loaded intersection polygon.')

        # Spatial join
        fishnet = fishnet.sjoin(gdfI, how = 'inner')

         # Drop any useless columns 
        fishnet = gpd.GeoDataFrame(geometry = fishnet['geometry'])

        if verbose == True:
            print('Kept only', len(fishnet), 'intersecting squares.')

    # Create centroids
    if centroid == True:

        centroids = fishnet.centroid

        if verbose == True:
            print('Created centroids.')

    # Output
    if output == 'gdf':
        if centroid == True:    
            return fishnet, centroids
        if centroid == False:
            return fishnet
        
    else:
        if centroid == True:

            if verbose == True:
                print('Exporting fishnet and centroids...')
            
            fishnet.to_file(output)

            # Some filename manipuilation
            split = output.split('.')
            split.insert(-1, '_pts.')
            outputCentroids = ''.join(split)
            centroids.to_file(outputCentroids)

        if centroid == False:

            if verbose == True:
                print('Exporting fishnet...')

            fishnet.to_file(output)

######################################################################################################################################################

# Get subsquares of a specified size from a fishnet (created by fishnet()).
def subfishnet(fishnet, size, output):

    """
    Parameters:
    fishnet (gdf): Geodataframe of fishnet to get subsquares from. 

    size (int): Size of fishnet squares. In same unit as projection. 
    - Should be a divisor of fishnet square size (e.g., 10000 divides 60 km2 tiles into 6x6)

    output (str): Filename of shapefile

    Returns:
    subfishnet (shp): Subfishnet polygons. 
    """

    # Blank geodataframe to fill using fishnet columns
    subfishnet = gpd.GeoDataFrame(columns = fishnet.columns, geometry = 'geometry')

    # Create subfishnet...
    for tile in tqdm(fishnet.index): # For each tile in fishnet

        # Get bounds 
        bounds = fishnet.geometry.loc[tile].bounds
        minX, minY, maxX, maxY = bounds # Components

        # Create subsquares based on size
        x, y = (minX, minY)
        subsquares = []

        while y < maxY:
            while x < maxX:
                geom = geometry.Polygon([(x, y), 
                                        (x, y + size), 
                                        (x + size, y + size), 
                                        (x + size, y), 
                                        (x,  y)])
                subsquares.append(geom)
                x += size
            x = minX
            y += size    

        # Geodataframe work
        subsquares = gpd.GeoDataFrame(subsquares, columns = ['geometry']) 
        subsquares['ID'] = tile
        subsquares['ID_sub'] = subsquares.index    

        # Add to subfishnet
        subfishnet = pd.concat([subfishnet, subsquares])  

    subfishnet = subfishnet.set_crs(fishnet.crs) # Set crs
    subfishnet = subfishnet.astype({'ID': 'int32', 'ID_sub': 'int32'})  # Clean dtypes

    # Save to shapefile
    subfishnet.to_file(output)

######################################################################################################################################################
# Spectral ###########################################################################################################################################
######################################################################################################################################################

# Filter bands to common set.
def bandBuilder(bands, sat):

    """
    Parameters:
    bands (list): Band names of interest. Supports: 'BLUE' (hls, s2), 'GREEN' (hls, s2), 'RED' (hls, s2), 'REDEDGE1' (s2), 'REDEDGE2' (s2), 
    'REDEDGE3' (s2), 'NIR0' (s2), 'NIR' (hls, s2), 'SWIR1' (hls, s2), 'SWIR2' (hls, s2). 
    - Note: NIR0 in s2 corresponds to B8 (10 m NIR band). NIR corresponds to B8A (20 m NIR band that spectrally corresponds to Landsat NIR band)

    sat (string): Satellite constellation being used. Supports: 'hls', 's2'. 

    Returns:
    bands (list): Ajusted band list(s)
    """

    if sat == 'hls': # HLS version

        # Remove Fmask if it exists (in alternative list to preserve bands for later)
        bands1 = bands.copy()

        if 'Fmask' in bands1:
            bands1.remove('Fmask')

        # Select bands
        bandsL30 = bands1.copy()
        bandsS30 = bands1.copy()

        # Align band names
        for band in range(len(bands1)):
            if bands1[band] == 'BLUE':
                bandsL30[band] = 'B02'
                bandsS30[band] = 'B02'
            elif bands1[band] == 'GREEN':
                bandsL30[band] = 'B03'
                bandsS30[band] = 'B03'
            elif bands1[band] == 'RED':
                bandsL30[band] = 'B04'
                bandsS30[band] = 'B04'
            elif bands1[band] == 'NIR':
                bandsL30[band] = 'B05'
                bandsS30[band] = 'B8A'
            elif bands1[band] == 'SWIR1':
                bandsL30[band] = 'B06'
                bandsS30[band] = 'B11'
            elif bands1[band] == 'SWIR2':
                bandsL30[band] = 'B07'
                bandsS30[band] = 'B12'
            else:
                raise Exception('Only BLUE, GREEN, RED, NIR, SWIR1 and SWIR2 band names supported.')        
            
        return bandsL30, bandsS30
    
    if sat == 's2': # S2 version

        # Remove SCL if it exists (in alternative list to preserve bands for later)
        bands1 = bands.copy()     

        if 'SCL' in bands1:
            bands1.remove('SCL')   

        # Select bands
        bandsS2 = bands1.copy()

        # ALign band names
        for band in range(len(bands1)):
            if bands1[band] == 'BLUE':
                bandsS2[band] = 'blue'
            elif bands[band] == 'GREEN':
                bandsS2[band] = 'green'
            elif bands1[band] == 'RED':
                bandsS2[band] = 'red'
            elif bands1[band] == 'REDEDGE1':
                bandsS2[band] = 'rededge1'
            elif bands1[band] == 'REDEDGE2':
                bandsS2[band] = 'rededge2'      
            elif bands1[band] == 'REDEDGE3':
                bandsS2[band] = 'rededge3'         
            elif bands1[band] == 'NIR0':
                bandsS2[band] = 'nir'   
            elif bands1[band] == 'NIR':
                bandsS2[band] = 'nir08'
            elif bands1[band] == 'SWIR1':
                bandsS2[band] = 'swir16'
            elif bands1[band] == 'SWIR2':
                bandsS2[band] = 'swir22'
            else:
                raise Exception('Only BLUE, GREEN, RED, REDEDGE1, REDEDGE2, REDEDGE3, NIR0, NIR, SWIR1 and SWIR2 band names supported.') 
                
        return bands, bandsS2           
######################################################################################################################################################
    
######################################################################################################################################################
# Temporal ###########################################################################################################################################
######################################################################################################################################################   

#  Remove bad images from time-series cube based on metadata.
def removeBadScenes(cube, sceneCloudThresh, verbose = True):

    """
    Parameters:
    cube (xarray): xarray time-series cube (e.g., from stackstac or odc-stac)

    sceneCloudThresh (int): Highest scene level cloud % to include (e.g., from metadata). 100 = no cloud cover filter at scene level. 

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube (xarray): Cleaned time-series cube
    """

    # Remove images with bad times (NaT)
    cube = cube.sel(time = ~np.isnat(cube.time)) # Keep only times with a time, maintains dtype

    if verbose == True:
        print('Removed images with NaT times (n = ' + str(cube.shape[0]) + ').')
    
    # Remove images with high scene-level cloud cover
    if sceneCloudThresh < 100: 
        cube = cube.sel(time = cube['eo:cloud_cover'].values <= sceneCloudThresh) # Keep times where clouds <= threshold, maintains dtype

        if verbose == True:
            print('Removed images above ' + str(sceneCloudThresh) + '% scene-level clouds (n = ' + str(cube.shape[0]) + ').')    

    return cube
######################################################################################################################################################
    
# Create same-day median composites from time-series cube.
def sameDayMerge(cube, form, con, bands, sat, verbose = True):

    """
    Parameters:
    cube (xarray): xarray time-series cube (e.g., from stackstac or odc-stac)

    form (string): cube form. 'spectral': spectral data cube (merge with mean). 'mask': mask data cube (merge with max). 

    con (string): Add coordinate representing the satellite constellation. If left blank (''), no new coordinate is added. 

    bands (list): Rename bands to this list. Only matters for spectral cubes. 

    sat (string): Satellite constellation being used. Supports: 'hls', 's2'. 

    verbose (boolean): Whether (true) or not (false) to print function status

    Returns:
    cube(s) (xarray): Daily composite time-series cube(s) (spectral, qa)
    """  

    # Add coordinate representing satellite constellation
    if con != '':
        cube = cube.assign_coords(constellation = con)  

    # Set HMS to 0s (i.e., same day = same ob)
    cube['time'] = cube['time'].dt.floor('1D')

    # Drop unneeded variables
    if sat == 'hls':
        cube = cube.drop_vars(['id', 'end_datetime', 'eo:cloud_cover', 'start_datetime', 'epsg', 'storage:schemes'], errors = 'ignore')
    if sat == 's2':
        cube = cube.drop_vars(['id', 'mgrs:latitude_band', 'proj:epsg', 's2:product_type', 's2:datatake_type', 'instruments', 'constellation', 
                            'mgrs:utm_zone', 's2:saturated_defective_pixel_percentage', 'gsd', 'title', 'raster:bands', 'common_name',
                            'center_wavelength', 'full_width_half_max', 'epsg', 's2:vegetation_percentage', 's2:reflectance_conversion_factor',
                            'processing:software', 's2:granule_id', 's2:degraded_msi_data_percentage', 'updated', 'view:sun_elevation',
                            's2:high_proba_clouds_percentage', 'earthsearch:s3_path', 's2:medium_proba_clouds_percentage', 
                            's2:snow_ice_percentage', 's2:processing_baseline', 's2:sequence', 'platform', 'mgrs:grid_square', 's2:product_uri',
                            's2:water_percentage', 'eo:cloud_cover', 's2:datastrip_id', 'earthsearch:payload_id', 'grid:code', 'created',
                            's2:cloud_shadow_percentage', 'view:sun_azimuth', 's2:nodata_pixel_percentage', 's2:datatake_id', 
                            's2:not_vegetated_percentage', 's2:dark_features_percentage', 'earthsearch:boa_offset_applied', 
                            's2:thin_cirrus_percentage', 's2:generation_time', 's2:unclassified_percentage', 's2:mgrs_tile'], errors = 'ignore') 
    
    # If same day observations need to be merged...
    if len(cube.groupby('time')) != len(cube):

        if form == 'spectral': # Create daily means
            #if sat == 'hls': # Lowest memory dtype may differ by sat
                #cube = cube.groupby('time').mean(engine = 'flox') # Mean on same day, flox faster (changes dtype + bad if NA a value (e.g., int16))
            cube = cube.groupby('time').apply(ss.mosaic, dim = 'time', nodata = -9999) # First ob (maintains dtype, slightly faster than flox)

        if form == 'mask':
            #if sat == 'hls': # Lowest memory dtype may differ by sat
                #cube = cube.groupby('time').max(engine = 'flox') # Max on same day, flox is faster (chooses method for us)
            cube = cube.groupby('time').apply(ss.mosaic, dim = 'time', nodata = 255) # First ob (maintains dtype, slightly faster than flox)

        if verbose == True:
            print('Same-day time-steps merged (n = ' + str(cube.shape[0]) + ').')  

    # Rename bands (only matters for spectral)
    if form == 'spectral':
        if 'Fmask' in bands or 'SCL' in bands: 
            cube['band'] = bands[0:len(bands) - 1]
        else:
            cube['band'] = bands   

    return cube
######################################################################################################################################################

# Clean cube by removing time-steps with high NA % OR high unclear % (from mask) and/or pixel masking based on mask.
def timestepClean(cube, mask = [], timestepClean = '',  thresh = 100, valid_status = 'all', invalid = 0, cleanMask = False, verbose = True): 

    """
    Parameters:
    cube (xarray): cube to mask.

    mask (xarray): mask to apply. Should be in form 1/True = do not mask (clear), 0/False = mask (unclear).
    - mask should have same time-steps as cube.
    - not required if just doing timestepClean = 'NA'.

    timestepClean (str): Type of time-step cleaning. '' = skip. 'NA': clean based on NA values. 'unclear': clean based on unclear values in mask.

    thresh (int): Highest NA or unclear % time-step to include. 100 = no removing.

    valid_status (str): How to calculate valid (clear + unclear) pixel count. Only applies if timestepClean == 'unclear'. 
    - 'all': All pixels in cube can be considered valid. Cube shape used to define valid count. 
    - 'static': Some pixels are invalid (e.g., outside ROI), but they are the same across all time-steps. First time-step used to define valid count.
    - 'dynamic': Valid pixel count may vary by time-step, so valid counts are calculated for all time-steps. 

    invalid (float): Value to be considered invalid (i.e., not clear or unclear, such as outside ROI). 
    - Only used if valid_status == 'static' or 'dynamic'. 

    cleanMask (bool): Whether (True) or not (False) to clean and output mask cube as well.

    verbose (bool): Whether (True) or not (False) to print function status.

    Returns:
    cube (xarray): cleaned cube.
    """

    # Timestep level cleaning
    if (timestepClean == 'NA') & (thresh < 100): # NA time-step cleaning

        if cube.ndim == 4: # Multiple bands (e.g., spectral cube)
            NAcount = ((cube.shape[-3] * cube.shape[-2] * cube.shape[-1]) * ((100 - thresh) / 100)) 
        if cube.ndim < 4: # Single band (e.g., qa cube)
            NAcount = ((cube.shape[-2] * cube.shape[-1]) * ((100 - thresh) / 100))  

        # Drop timesteps with few clear pixels
        cube = cube.dropna(dim = 'time', thresh = NAcount) # Some weirdness with dropna and duplicate dates...

        if (cleanMask == True) & (type(mask) != list):
            mask = mask.dropna(dim = 'time', thresh = NAcount)  # Some weirdness with dropna and duplicate dates...

        if verbose == True:
            print('Removed time-steps above ' + str(thresh) + '% NA pixels (n = ' + str(cube.shape[0]) + ').')      

    if (timestepClean == 'unclear') & (thresh < 100): # unclear time-step cleaning (based on provided mask)

        # If mask does not have same time-steps as cube, raise error
        if np.array_equal(cube.time.values, mask.time.values) == False:
            raise Exception('Time-step mismatch between cube and mask.')    

        # Get valid count based on valid_status
        if valid_status == 'all':
            valid = cube.shape[-2] * cube.shape[-1] # All pixels in cube are valid
        if valid_status == 'static':
            valid = xr.where(cube[0] != invalid, 1, 0).sum(dim = ['x', 'y']) # Invalid pixels don't change through time (calc from first)
        if valid_status == 'dynamic':
            valid = xr.where(cube != invalid, 1, 0).sum(dim = ['x', 'y']) # Valid pixel count for each time-step
            
        # Get clear count for each timestep
        clear = xr.where(mask, 1, 0).sum(dim = ['x', 'y']) # Number of clear pixels for each time-step in array

        # Filter cube based on clear and valid counts for each time-step 
        boolArr = (clear / valid * 100) > 100 - thresh
        cube = cube.sel(time = boolArr) # Keep only time-steps below unclear thresh

        if cleanMask == True:
            mask = mask.sel(time = boolArr)

        if verbose == True:
            print('Removed time-steps above ' + str(thresh) + '% unclear pixels (n = ' + str(cube.shape[0]) + ').')

    if cleanMask == False:
        return cube
    if cleanMask == True:
        return cube, mask

######################################################################################################################################################
