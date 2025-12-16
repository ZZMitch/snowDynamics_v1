######################################################################################################################################################
#
#   name:       TrainingData_Utils.py
#   contains:   Functions for preparing training data
#   created by: Mitchell Bonney
#
#####################################################################################################################################################

import geopandas as gpd
import glob
import numpy as np
import os
import pandas as pd
import rasterio as rio

from rasterio import features
from shapely.geometry import Polygon
from tqdm.notebook import tqdm

# Recommended to have installed...
# https://pyogrio.readthedocs.io/en/latest/ 

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Condsider shapefile alternatives, although generally polygons used are simple (e.g., https://geoparquet.org/)

######################################################################################################################################################
# Sample Locations ###################################################################################################################################
######################################################################################################################################################

# Get valid (i.e., not nodata) footprint from a raster image (e.g., high-res image from which to build training data)
def validFootprint(tif, zeroValid, buffer, download, verbose):

    """
    Parameters:
    tif (string): Location of tif file ('C:/path/to/file.tif'). Tif should be in a projected coordinate system with m units. 

    zeroValid (boolean): Whether (true) or not (false) zero (0) is considered a valid value (in addition to nodata value(s))

    buffer (int): Buffer created shapefile in meters (e.g., -100 to accounted for edge effects in image). 0 = no buffering. 
    Absolute value also used for simplying polygon. 

    download (string): Location to save output shapefile ('C:/path/to/file.shp')

    verbose (boolean): Whether (true) or not (false) to print function status

    Results:
    shapefile of valid data in image
    """

    # Load file within rasterio
    with rio.open(tif) as file:
        img = file.read(1) # Just 1st band

        if verbose == True:
            print('Image loaded.')

        # In image
        #nodata = np.argmax(np.bincount(np.reshape(img, img.size))) # Reshape to 1D, get count for each value, get value with max count        
            
        # If there is no recorded nodata value...
        if file.nodata == None:
            
            if verbose == True:
                print('No recorded nodata value.')
                
            # Most common value along edges, looking 5 in (very likely to be non-valid, "black" values)
            top = img[0:5,:].flatten()
            right = img[1:,-5:].flatten()
            bottom = img[-5:,:-1].flatten()
            left = img[1:-1,0:5].flatten()               

            edges = np.concatenate([top, right, bottom, left])
            black = np.argmax(np.bincount(edges)) 
            
            if verbose == True:
                print('Additional non-valid value:', black) 
            
            # Create binary image (0 = nodata, 1 = valid)
            if zeroValid == True:
                img = (img != black).astype(np.uint8) # Just black value set as non-valid   
            if zeroValid == False:
                img = ((img != black) & (img != 0)).astype(np.uint8)     
        
        # If there is a nodata record...  s
        if file.nodata != None:    
            
            # Recorded nodata value
            nodata = int(file.nodata)

            if verbose == True:
                print('Recorded nodata value:', nodata)
                
            # Most common value along edges that is not nodata (very likely to be non-valid, "black" values)
            top = img[0:5,:].flatten()
            right = img[1:,-5:].flatten()
            bottom = img[-5:,:-1].flatten()
            left = img[1:-1,0:5].flatten()               

            edges = np.concatenate([top, right, bottom, left])
        
            bincount = np.bincount(edges)
            bincount[nodata] = 0 # Set nodata bincount to 0 (in case > black)
            
            black = np.argmax(bincount) # Value of largest non-nodata

            if verbose == True:
                print('Additional non-valid value:', black)     

            # Create binary image (0 = nodata, 1 = valid)
            if zeroValid == True:
                img = ((img != black) & (img != nodata)).astype(np.uint8) # Both values set as non-valid
            if zeroValid == False: 
                img = ((img != black) & (img != nodata) & (img != 0)).astype(np.uint8)

        if verbose == True:
            print('Binary image created.')

        # Vectorize value polygon
        shapes = features.shapes(img, transform = file.transform)
        polys = [Polygon(shape[0]['coordinates'][0]) for shape in shapes if shape[1] == 1]
        footprint = gpd.GeoDataFrame(crs = file.crs, geometry = polys)

        # If more than one polygon, just select ones larger than 1 km2 (1000000 m2) and make multipoly
        if len(footprint) > 1:
            #footprint = footprint.iloc[[footprint.area.idxmax()]] # Does not work for images with nodata gaps between valid regions
            footprint = footprint[footprint.area > 1000000].dissolve()

        if verbose == True:
            print('Footprint created.')

        # If requested, buffer (and simplify, which saves significant processing time and reduces shp size x100)
        if buffer != 0: 
            footprint['geometry'] = footprint['geometry'].simplify(abs(buffer)).buffer(buffer)

            if verbose == True:
                print('Footprint buffered.')

        # Export as shp
        footprint.to_file(download, engine = 'pyogrio')

        if verbose == True:
            print('Shapefile created.')

# For all images in a folder structure, get valid footprint
def validFootprints(path, zeroValid, buffer, verbose):

    """
    Parameters:
    path (string): Location of tifs in folder structure. Will find tifs interior to folders within this folder. ('C:/path/to/tifs')

    zeroValid (boolean): Whether (true) or not (false) zero (0) is considered a valid value (in addition to nodata value(s))

    buffer (int): Buffer created shapefile in meters (e.g., -100 to accounted for edge effects in image). 0 = no buffering. 

    verbose (boolean): Whether (true) or not (false) to print function status

    Results: 
    Shapefiles of valid data for all images in same folder as each image
    """

    # Get path to all tifs in folder structure
    tifs = glob.glob(os.path.join(path, '**', '*.tif'), recursive = True) # ** and recursive allow for internal folder searching

    if verbose == True:
        print('Found', len(tifs), 'tifs in this folder structure.')

    # For each tif, create shapefile of valid data extent and store in same folder as tif
    for tif in tqdm(tifs):

        print('Creating shp for', tif.rsplit('\\', 1)[-1], '...')

        # Create shp path
        folder = tif.rsplit('\\', 1)[0] # Everything before tif file
        ID = folder.rsplit('\\', 1)[-1] # ID number of tif file 
        shp = os.path.join(folder, str(ID) + '_Footprint.shp')

        # Generate footprint (if needed)
        if os.path.exists(shp): # If shapefile already created, skip
            print('Footprint shp already created.')

        else: 
            validFootprint(tif, zeroValid, buffer, shp, verbose)
######################################################################################################################################################

# Find all footprint shps in a folder structure, merge together and clip to specific AOIs        
def mergeFootprints(polys, aoi, metadata, clip, download, verbose):

    """
    Parameters:
    polys (string): Location of footprint polys in folder structure. Will find shps interior to folders within this folder. ('C:/path/to/polys')

    aoi (string): Location of shp representing area of interest (e.g., study area). Used to specify global crs for all footprints

    metadata (string): Location of metadata csv. '' = no metadata.
    - Needs an 'ID' field that matches identifier on footprints file name. 
    - Needs a 'Date' field to convert to datetime

    clip (string): How to clip footprints. '' = no clipping. 'aoi' = clip by area of interest. 

    download (string): Location to save output shapefile ('C:/path/to/file.shp')

    verbose (boolean): Whether (true) or not (false) to print function status

    Results:
    Shapefile with clipped polygons representing potential sample areas
    """

    # Get path of all footprints in folder structure
    polys = glob.glob(os.path.join(polys, '**', '*_Footprint.shp'), recursive = True) # ** and recursive allow for internal folder searching

    if verbose == True:
        print('Found', len(polys), 'footprints in this folder structure.')

    # Load aoi
    aoi = gpd.read_file(aoi, engine = 'pyogrio')
   
    # Merge into one GeoDataFrame
    gdf_list = []
    for poly in polys: # For each footpring path string
        ID = int(poly.split('\\')[-1].split('_')[0]) # Get ID from file name
        gdf = gpd.read_file(poly, engine = 'pyogrio', columns = []).to_crs(aoi.crs).assign(ID = ID) # Load shp, reproject, assign ID column
        gdf_list.append(gdf) # Add to list
         
    footprints = pd.concat(gdf_list).pipe(gpd.GeoDataFrame) # Combine list and pipe to gdf
    # footprints = pd.concat([gpd.read_file(poly, engine = 'pyogrio',columns = []).to_crs(aoi.crs) for poly in polys]).pipe(gpd.GeoDataFrame)

    # Note: The code below being grayed out is a bug with Visual Studio, code works fine
    if verbose == True:
        print('Footprints merged and projected (based on AOI).')

    # Add metadata
    if metadata != '':
        metadata = pd.read_csv(metadata)

        # Add columns to footprints based on matching 'ID' columns
        footprints = footprints.merge(metadata, on = 'ID')

        # Convert 'Date' Field to datetime
        footprints['Date'] = pd.to_datetime(footprints['Date'])

        if verbose == True:
            print('Added columns based on metadata file.')

    # Clip by aoi
    if clip == 'aoi':
        footprints = gpd.clip(footprints, aoi)

        if verbose == True:
            print('Footprints clipped to AOI.')

    # Export as shp
    footprints.to_file(download, engine = 'pyogrio')

    if verbose == True:
        print('Shapefile created.')