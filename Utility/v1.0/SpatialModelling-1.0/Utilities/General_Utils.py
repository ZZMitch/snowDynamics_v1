######################################################################################################################################################
#
#   name:       General_Utils.py
#   contains:   Functions for general processing of geospatial data. 
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Open source
import xvec
import numpy as np

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Add more functionality to cubeZonal() - More statistics, check to confirm zones and cube are same projection, support for continuous cubes. 

######################################################################################################################################################
# Data Cube Processing ###############################################################################################################################
######################################################################################################################################################

# Apply zonal statistics on a data cube time-series and output a dataframe
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