######################################################################################################################################################
#
#   name:       General_Utils.py
#   contains:   Functions for general processing of geospatial data. 
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Built in
import math

# Open source
import xvec
import numpy as np
import geopandas as gpd
import pygeoops

from shapely.geometry import LineString, Point

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Add more functionality to cubeZonal() - More statistics, check to confirm zones and cube are same projection, support for continuous cubes. 

######################################################################################################################################################
# Data Cube Processing ###############################################################################################################################
######################################################################################################################################################

# Apply zonal statistics on a data cube time-series and output a dataframe.
def cubeZonal(cube, 
              zones,
              form = 'discrete', 
              stat = 'pCon', 
              condition = [], 
              direction = '', 
              valid = [], 
              verbose = True):

    """
    Parameters:
    cube (dataArray): Cube to apply zonal statistics on. 

    zones (gdf): Geodataframe defining zones (polygons). Needs to be same CRS as cube. 

    cube form (str): 'discrete' (e.g., land cover), 'continuous' (e.g., % cover)
    - Currently supports discrete only

    stat (str): Statistic to calculate. Supports: 'pCon'
    - pCon: Percent of zone that meets a condition. 

    condition (num or list of int): Pixel values that meet condition. Will be used to create conditional boolean cube. Applies to 'pCon'. 
    - Discrete: (list), Continuous (num)

    direction (str): '>', '>=', '<', '<='. Not yet supported. For continuous only. Only applies if len(condition) == 1.  
    - '' = No direction (discrete)

    valid (num, range of nums, or list of int): All valid pixel vaues (e.g., inside AOI). Will be used to define full area to calculate percentages.
    - Discrete: (list), Continuous (num) 

    verbose (bool): Whether (true) or not (false) to print function status  

    Returns:
    Pandas dataframe with statisic for each time-step. 
    """

    if form != 'discrete':
        raise Exception('Only discrete cubes currently supported.')   
    if stat != 'pCon':
        raise Exception('Only pCon currently supported.') 
    if direction != '':
        raise Exception('Direction not currently currently supported.') 
    
    # Get X and Y coords for zonal statistics (can add suport for other dimension names later)
    if 'x' in cube.dims:
        x = 'x'
    if 'y' in cube.dims:
        y = 'y'
    
    # Define conditional cube
    conCube = cube.isin(condition)

    if verbose == True:
        print('Converted cube to boolean (1 = Meets condition, 0 = Does not meet condition).')

    # Define valid pixel layer
    vPixels = cube[0,:,:].isin(valid) # Only need one time-step

    # Get number of valid pixels
    nPixels = vPixels.xvec.zonal_stats(zones.geometry, x, y, stats = 'sum', index = True, method = 'rasterize', all_touched = False).values
    #nPixels = int(vPixels.sum().values) # Only makes sense if full area covered by single zone

    if verbose == True:
        print('Calculated number of valid pixels for all zones (pixels = ' + str(int(np.nansum(nPixels))) + ', zones = ' + str(len(nPixels)) + ').')

    # if int(np.isnan(nPixels).sum()) > 0: # If some zones have no valid pixels (don't need this actually, will just get NaN columns in df)

    #     zones = zones[np.isnan(nPixels) == False] # Remove those zones from analysis

    #     if verbose == True:
    #         print('Removed ' + str(int(np.isnan(nPixels).sum())) + ' zones with 0 valid pixels.')

    # Calculate number of pixels that meet condition (pCon) for all zones
    nCon = conCube.xvec.zonal_stats(zones.geometry, x, y, stats = 'sum', index = True, method = 'rasterize', all_touched = False) #.values
    # Outputs dataArray with geometry = len(zones) and time = len(cube)

    # Rename index with index of zone

    if verbose == True:
        print('Calculated number of pixels that met condition for all zones (' + str(nCon.shape[0]) + 
              ') and time-steps (' + str(nCon.shape[1]) + ').')
        
    # Fill in dataframe with pCon (nCon / nPixels for all zones and time-steps). Use zones index for column names. 
    df = nCon.swap_dims({'geometry': zones.index.name}).T.to_pandas() # Transpose is required because otherwise time ends up as columns
    df = df / nPixels * 100 # Convert to pCon

    if verbose == True:
        print('Created dataframe where each column is % pixels meeting condition across all time-steps for each zone.')

    return df

######################################################################################################################################################
# Vector Processing ##################################################################################################################################
######################################################################################################################################################

# Centerline function that always returns a single line-string (e.g., rivers).
def centerline_single(gdf, simplify_poly = 30, densify_distance = 30, min_branch_length = 100, min_branch_iter = 100, 
                      simplify_centerline = 0, extend = False, verbose = True):

    """
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame containing linear polygon geometries. Function will cycle through each vector. 

    simplify_poly (float): Simplify input polygons, which will speed up processing. 
    - Found 30 works well for vectors based on 30 m sized pixels (e.g., match this to pixel size)
    - 0: No simplification
    - For more details, see: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html 

    densify_distance (float): Densify input geometry so each segment has maximum this length. 
    - Found 30 works well for vectors based on 30 m sized pixels (e.g., match this to pixel size)
    - 0: No densification
    - For more details, see: https://pygeoops.readthedocs.io/en/latest/api/pygeoops.centerline.html 

    min_branch_length (float): Minimum length for branches of the main centerline. 
    - In units of crs (e.g., 100 = 100 m minimum). Setting very high can remove portions of main channel. 
    - Must be > 0
    - For more details, see: https://pygeoops.readthedocs.io/en/latest/api/pygeoops.centerline.html  

    min_branch_iter (float): How much to iterate min_branch_length by to find optimal output
    - In units of crs (e.g., 100 = 100 m). Smaller values increase processing time. 
    - 0 = Not iteration. 

    simplify_centerline (float): Tolerance to simplify the resulting centerline. 
    - 0 (default) is no simplification, which is most accurate if processing speed is not an issue
    - For more details, see (simplify_tolerance): https://pygeoops.readthedocs.io/en/latest/api/pygeoops.centerline.html

    extend (bool): Extend the centerline to the edge of the geometry.
    - Default is False.
    - For more details, see: https://pygeoops.readthedocs.io/en/latest/api/pygeoops.centerline.html

    verbose (bool): Whether (true) or not (false) to print function status.  

    Returns: 
    GeoDataFrame: A GeoDataFrame containing the centerline linestrings.
    """

    min_branch_length_default = min_branch_length # Save setting

    # Simplify geodataframe
    if simplify_poly > 0:
        gdf['geometry'] = gdf.geometry.simplify(simplify_poly)

        if verbose == True:
            print('Simplified geodataframe polygons.')

    # Create gdf copy to fill with centerlines
    cl_gdf = gdf.copy()
    cl_gdf['geometry'] = None

    # For each geometry...
    for geom in gdf.geometry:

        if verbose == True:
            print('--------------------')
            print(f'...Polygon {gdf.index.get_loc(gdf[gdf.geometry == geom].index[0]) + 1} of {len(gdf)}...')
        
        # Run centerline function (pygeoops.centerline) with default settings
        cl = pygeoops.centerline(geom, densify_distance = densify_distance, 
                                 min_branch_length = min_branch_length, 
                                 simplifytolerance = simplify_centerline, 
                                 extend = extend)
        
        if verbose == True:
            print('Ran centerline function with supplied parameters.')

        # Keep running centerline (iterating min_branch_length) until 1 line per input polygon (i.e., no branches)
        cl_len = 1 if cl.geom_type == 'LineString' else len(cl.geoms) # Get number of lines
        geom_len = 1 if geom.geom_type == 'Polygon' else len(geom.geoms) # Get number of polygons

        while cl_len > geom_len:

            # Update min_branch_length by adding iteration value
            min_branch_length += min_branch_iter
        
            if verbose == True:
                print(str(cl_len - geom_len) + ' unwanted branches found, re-running with min_branch_length = ' + str(min_branch_length) + '.')

            # Run centerline function (pygeoops.centerline) with updated settings
            cl = pygeoops.centerline(geom, densify_distance = densify_distance, 
                                     min_branch_length = min_branch_length, 
                                     simplifytolerance = simplify_centerline, 
                                     extend = extend)
            cl_len = 1 if cl.geom_type == 'LineString' else len(cl.geoms) # Get number of lines

        if verbose == True:
            print('Created ' + str(cl.geom_type) + ' centerline from input polygon.')
        min_branch_length = min_branch_length_default # Reset

        # If input is a MultiPolygon (e.g., there are gaps in linear polygons), connect nearest points in centerline
        if geom_len > 1:
            cl = merge_mls(cl, verbose)

        # Add to geodataframe
        cl_gdf.loc[gdf[gdf.geometry == geom].index[0], 'geometry'] = cl

        if verbose == True:
            print('Added centerline to geodataframe.')

    return cl_gdf

######################################################################################################################################################

# Merge a MultiLineString into a single LineString by iteratively connecting nearest endpoints.
def merge_mls(mls, verbose = True):

    """
    Parameters:
    mls (MultiLineString): The input MultiLineString to merge.

    verbose (bool): Whether (true) or not (false) to print function status.  

    Returns:
    LineString: A single LineString created by merging the input MultiLineString.
    """

    if mls.geom_type == "LineString":
        return mls
    if mls.geom_type != "MultiLineString":
        raise ValueError("Input must be LineString or MultiLineString")

    # Start with the first line
    lines = list(mls.geoms)
    merged = lines.pop(0)

    while lines:
        # Find the nearest line by endpoint distance
        nearest_idx = None
        nearest_dist = math.inf
        reverse_line = False
        append_to_start = False

        for i, ln in enumerate(lines):
            dists = [(Point(merged.coords[-1]).distance(Point(ln.coords[0])), "end_start"),
                     (Point(merged.coords[-1]).distance(Point(ln.coords[-1])), "end_end"),
                     (Point(merged.coords[0]).distance(Point(ln.coords[0])), "start_start"),
                     (Point(merged.coords[0]).distance(Point(ln.coords[-1])), "start_end"),]

            min_d, relation = min(dists, key=lambda x: x[0])

            if min_d < nearest_dist:
                nearest_dist = min_d
                nearest_idx = i
                reverse_line = relation in ["end_end", "start_start"]
                append_to_start = relation in ["start_start", "start_end"]

        # Pull out nearest line
        nearest = lines.pop(nearest_idx)
        if reverse_line:
            nearest = LineString(list(nearest.coords)[::-1])

        # Merge into existing line
        if append_to_start:
            merged = LineString(list(nearest.coords) + list(merged.coords))
        else:
            merged = LineString(list(merged.coords) + list(nearest.coords))

    if verbose == True:
        print('Converted MultiLineString to LineString by connecting nearest points')

    return merged

######################################################################################################################################################