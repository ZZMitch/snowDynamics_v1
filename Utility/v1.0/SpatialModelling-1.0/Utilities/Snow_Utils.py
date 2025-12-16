######################################################################################################################################################
#
#   name:       Snow_Utils.py
#   contains:   Functions for creating snow products
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Built in
import sys
import math
import datetime as dt
import os
import glob

# Open source
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from rasterio.enums import Resampling
#from scipy.signal import find_peaks

# Mine
sys.path.append('C:/Users/mbonney/OneDrive - NRCan RNCan/Projects/UtilityCode/DataAccess/Utilities')
import PreProcess_Utils as pputil

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Add alternative to split_date procedure for uncertainty in dailySnowCube2SnowDynamics() using scipy find_peaks or user supplied from IMS
# mergeTiledSnowDynamics() outputs warnings about large Dask graph sizes (using warnings ignore does not remove, happens on Dask end I think)
# implausible_snow in cleanSnowCube() may remove real snow observations sometimes (e.g., mountains, lakes) if small patches are not captured by IMS
# Test impact of xr.set_options(use_bottleneck=False) # For context, got wrong std values in testing... this fixes (but maybe slightly slower?) 
# Can reproject with dask: https://corteva.github.io/rioxarray/html/rioxarray.html (see links here)

######################################################################################################################################################
# HLS ################################################################################################################################################
######################################################################################################################################################

# Converts annual Fmask 10-cat cubes from observationAvailabilityHLS() to snow cubes where snow = 1, no snow = 0, clouds etc. = NaN.
def annualFmask2SnowCube(cube_yr1, cube_yr2, yrs = 2, doy = '', verbose = True):

    """
    Parameters:
    cube_yr1 (dask dataArray): Fmask cube (e.g., from observationAvailabilityHLS(), buildHLS()). Year will represent the first portion of SnowCube. 
    - fill (10) > cloud (9) > shadow (8) > cloud adjacent (7) > aerosol (snow: 6, water: 5, land: 4) > snow (3) > water (2) > land (1)

    cube_yr2 (dask dataArray): Fmask cube (e.g., from observationAvailabilityHLS(), buildHLS()). Year will represent the last portion of SnowCube. 
    - fill (10) > cloud (9) > shadow (8) > cloud adjacent (7) > aerosol (snow: 6, water: 5, land: 4) > snow (3) > water (2) > land (1) 

    yrs (int): Length (in years) of Fmask snow cube. Supports 1 and 2. 
    - 1: Uses user defined day-of-year ('doy') to create cube. E.g., Winter year boundary defined by HLS processing tile from IMS. 
    - 2: Creates a full 2-year cube (January 1 to December 31). Beneficial for HLS to help find clear observations before snow fall.  

    doy (int): Day of year to split annual cubes for creating 1-year snow cube. Only applies if yrs = 1
    - Starts cube day after doy and ends on doy next year. 

    verbose (bool): Whether (true) or not (false) to print function status  

    Returns:
    Cube of set length where snow = 1, non-snow = 0, unclear = NaN. 
    """

    if verbose == True:
        print('Initial Fmask time-steps: Start (n = ' + str(len(cube_yr1)) + '), End (n = ' + str(len(cube_yr2)) + ').')

    if yrs == 1:
        yr2 = int(cube_yr2.time.dt.year[-1]) # Last year represented in cube

        # Convert day-of-year to filterable strings
        if ((yr2 - 1) / 4).is_integer() == False: # Not a leap year
            date1 = dt.datetime(int(cube_yr1.time.dt.year[0]), 1, 1) + dt.timedelta(doy) # Day after first year
        if ((yr2 - 1) / 4).is_integer() == True: # Leap year
            date1 = dt.datetime(yr2 - 1, 1, 1) + dt.timedelta(doy + 1) # Day after first year
        if (yr2 / 4).is_integer() == False: # Not a leap year (doy )
            date2 = dt.datetime(yr2, 1, 1) + dt.timedelta(doy - 1) # Day-of-year next year
        if (yr2 / 4).is_integer() == True: # Leap year
            date2 = dt.datetime(yr2, 1, 1) + dt.timedelta(doy) # Day-of-year next year

        start = f'{date1.year}-{date1.month:02d}-{date1.day:02d}'
        end =  f'{date2.year}-{date2.month:02d}-{date2.day:02d}'

        # Filter dates
        cube_yr1 = cube_yr1.sel(time = slice(start, str(cube_yr1.time.dt.year[0]) + '-12-31'))
        cube_yr2 = cube_yr2.sel(time = slice(str(cube_yr2.time.dt.year[0]) + '-01-01', end))

        if verbose == True:
            print('Filtered to winter year (' + start + ' to ' + end + 
                  '): Start (n = ' + str(len(cube_yr1)) + '), End (n = ' + str(len(cube_yr2)) + ').')

    # On same day, different sensor, observations, choose minimum value, skipping NA (e.g., land > fill)
    cube_yr1 = cube_yr1.groupby('time').min(engine = 'flox', skipna = True).squeeze() # Don't need band dimension 
    cube_yr2 = cube_yr2.groupby('time').min(engine = 'flox', skipna = True).squeeze() # Don't need band dimension 

    if verbose == True:
        print('Made decision on same-day observations (L30vS30): Start (n = ' + str(len(cube_yr1)) + '), End (n = ' + str(len(cube_yr2)) + ').')

    # Combine year 1 and year 2 cube along time dimension
    snowCube = xr.concat((cube_yr1, cube_yr2), dim = 'time').sortby('time')#.squeeze() 
    #snowCube = pputil.loadXR(snowCube) # Loads into memory, faster without this...

    if verbose == True:
        print('Combined snow cube (n = ' + str(len(snowCube)) + ').')

    # Keep only time-steps that have 1%+ good values
    snowCube = pputil.timestepClean(snowCube, snowCube.isin([1,2,3,4,5,6]), timestepClean = 'unclear', valid_status = 'static', invalid = 0,
                                    thresh = 99, verbose = verbose)

    # Reclassify to snow = 1, non-snow ground observations = 0, unclear = NaN
    crs = snowCube.rio.crs # xarray where removes crs info... 

    snowCube = xr.where((snowCube == 0) | (snowCube >= 7), np.nan, snowCube).astype('float32') # Unclear (0, 10 - 7) = NaN
    snowCube = xr.where((snowCube == 1) | (snowCube == 2) | (snowCube == 4) | (snowCube == 5), 0, snowCube) # Non-snow ground observations (1, 2, 4, 5) = 0
    snowCube = xr.where((snowCube == 3) | (snowCube == 6), 1, snowCube) # Snow (3, 6) = 1
    snowCube.rio.write_crs(crs, inplace = True) # Reapply crs

    if verbose == True:
        print('Reclassifed (Snow = 1, Non-snow = 0, Unclear = NaN)')

    return snowCube

######################################################################################################################################################

# Converts HLS snow cube (e.g., from annualFmask2SnowCube()) to output Dataset quantifying clear observations per day for various temporal windows.
def clearObservationsPerDay_WinterYear(snowCube, windows = ['winterYear', 'snowSeason', 'snowFall', 'snowPeak', 'snowMelt'], smin_doy = '', 
                                       smax_doy = '', sstart_doy = '', send_doy = '', verbose = True):

    """
    Parameters:
    snowCube (xarray dataArray): xarray snow cube (e.g., output from annualFmask2SnowCube()). 

    windows (list of str): Temporal windows to create products for. Supports: 'winterYear', 'snowSeason', 'snowFall', 'snowPeak', 'snowMelt'.
    - winterYear: Day after snow minimum to next year snow minimum. Number of days should be 365 or 366. Requires smin_doy. 
    - snowSeason: One week before snow start to one week after snow end. Number of days variable. Requires sstart_doy and send_doy.
    - snowFall: One week before snow start to six weeks after snow start. Number of days should be 49. Requires sstart_doy.
    - snowPeak: Seven weeks around snow maximum. Number of days should be 49. Requires smax_doy.
    - snowMelt: Six weeks before snow end to one week after snow end. Number of days should be 49. Requires send_doy. 

    smin_doy (int): Day of year (julian day) representing minimum snow coverage. Must be positive.

    smax_doy (int): Day of year (julian day) representing maximum snow coverage. Must be positive.

    sstart_doy (int): Day of year (days since Dec 31 of winterYear) representing snow start. Can be negative (i.e., before Dec 31).

    send_doy (int): Day of year (days since Dec 31 of winterYear) representing snow end. Can be negative (i.e., before Dec 31).

    verbose (bool): Whether (true) or not (false) to print function status.

    Returns: 
    clearObsPerDay: Clear observations per day produts (xarray Dataset).
    """

    # Create empty xarray Dataset to fill with selected products
    crs = snowCube.rio.crs # Some xarray functions remove crs 
    clearObsPerDay = xr.Dataset(coords = dict(x = ('x', snowCube['x'].values), y = ('y', snowCube['y'].values)))
    clearObsPerDay.rio.write_crs(crs, inplace = True) # Reapply crs

    if verbose == True:
        print('Created empty clearObsPerDay Dataset to fill.')  

    if 'winterYear' in windows: # Clear observations per day for winter year

        # Define winter year start from snow minimum date
        wys_date = dt.datetime(int(snowCube.time.dt.year[0]), 1, 1) + dt.timedelta(smin_doy) # Day after first year
        wys_str = f'{wys_date.year}-{wys_date.month:02d}-{wys_date.day:02d}' # Filterable date

        # Define winter year end from snow minimum date
        wye_date = dt.datetime(int(snowCube.time.dt.year[-1]), 1, 1) + dt.timedelta(smin_doy - 1) # Day-of-year next year
        wye_str =  f'{wye_date.year}-{wye_date.month:02d}-{wye_date.day:02d}' # Filterable date

        # Get observations per day for winter year
        opd_wy = snowCube.sel(time = slice(wys_str, wye_str)) # Filter to winter year
        opd_wy = opd_wy.notnull().sum(dim = 'time') / ((wye_date - wys_date).days + 1) # Count clear, divide by number of possible days
        opd_wy.rio.write_crs(crs, inplace = True) # Reapply crs
        clearObsPerDay['winterYear_clearObsPerDay'] = opd_wy.astype('float32') # Add to clearObsPerDay

        if verbose == True:
            print('Added winter year (' + wys_str + ' to ' + wye_str + ') to clearObsPerDay.')

    # Clear observations per day for snow season (dates also help snowFall and snowMelt)
    if ('snowSeason' in windows) | ('snowFall' in windows): # Define snow season start from snow start date
        sss_date = dt.datetime(int(snowCube.time.dt.year[0]), 12, 31) + dt.timedelta(sstart_doy - 7) # One week before snow start
        sss_str = f'{sss_date.year}-{sss_date.month:02d}-{sss_date.day:02d}'

    if ('snowSeason' in windows) | ('snowMelt' in windows): # Define snow season end from snow end date
        sse_date = dt.datetime(int(snowCube.time.dt.year[0]), 12, 31) + dt.timedelta(send_doy + 7) # One week after snow end
        sse_str =  f'{sse_date.year}-{sse_date.month:02d}-{sse_date.day:02d}'

    if 'snowSeason' in windows: # Get observations per day for snow season
        opd_ss = snowCube.sel(time = slice(sss_str, sse_str)) # Filter to snow season
        opd_ss = opd_ss.notnull().sum(dim = 'time') / ((sse_date - sss_date).days + 1) # Count clear, divide by number of possible days
        opd_ss.rio.write_crs(crs, inplace = True) # Reapply crs
        clearObsPerDay['snowSeason_clearObsPerDay'] = opd_ss.astype('float32') # Add to clearObsPerDay

        if verbose == True:
            print('Added snow season (' + sss_str + ' to ' + sse_str + ') to clearObsPerDay.') 

    if  'snowFall' in windows: # Clear observations per day for snow fall period

        # Define snow fall end from snow start date (snow fall start already defined)
        sfe_doy = sstart_doy + 41 # Snow fall end: 6 weeks after snow start
        sfe_date = dt.datetime(int(snowCube.time.dt.year[0]), 12, 31) + dt.timedelta(sfe_doy)
        sfe_str =  f'{sfe_date.year}-{sfe_date.month:02d}-{sfe_date.day:02d}'

        # Get observations per day for snow fall period
        opd_sf = snowCube.sel(time = slice(sss_str, sfe_str)) # Filter to snow fall period
        opd_sf = opd_sf.notnull().sum(dim = 'time') / ((sfe_date - sss_date).days + 1) # Count clear, divide by number of possible days
        opd_sf.rio.write_crs(crs, inplace = True) # Reapply crs
        clearObsPerDay['snowFall_clearObsPerDay'] = opd_sf.astype('float32') # Add to clearObsPerDay     

        if verbose == True:
            print('Added snow fall period (' + sss_str + ' to ' + sfe_str + ') to clearObsPerDay.') 

    if 'snowPeak' in windows: # Clear observations per day for snow peak period

        # Define snow peak start from snow maximum date
        sps_date = dt.datetime(int(snowCube.time.dt.year[0]), 12, 31) + dt.timedelta(smax_doy) - dt.timedelta(24) # 3.5 weeks before  
        sps_str = f'{sps_date.year}-{sps_date.month:02d}-{sps_date.day:02d}'

        # Define snow peak end from snow maximum date
        spe_date = dt.datetime(int(snowCube.time.dt.year[0]), 12, 31) + dt.timedelta(smax_doy) + dt.timedelta(24) # 3.5 weeks after
        spe_str =  f'{spe_date.year}-{spe_date.month:02d}-{spe_date.day:02d}'

        # Get observations per day for snow peak period
        opd_sp = snowCube.sel(time = slice(sps_str, spe_str)) # Filter to snow peak period 
        opd_sp = opd_sp.notnull().sum(dim = 'time') / ((spe_date - sps_date).days + 1) # Count clear, divide by number of possible days
        opd_sp.rio.write_crs(crs, inplace = True) # Reapply crs
        clearObsPerDay['snowPeak_clearObsPerDay'] = opd_sp.astype('float32') # Add to clearObsPerDay   

        if verbose == True:
            print('Added snow peak period (' + sps_str + ' to ' + spe_str + ') to clearObsPerDay.') 

    if  'snowMelt' in windows: # Clear observations per day for snow fall period

        # Define snow melt start from snow end date (snow melt start already defined)
        sms_doy = send_doy - 41 # Snow melt start: 6 weeks before snow end
        sms_date = dt.datetime(int(snowCube.time.dt.year[0]), 12, 31) + dt.timedelta(sms_doy)
        sms_str =  f'{sms_date.year}-{sms_date.month:02d}-{sms_date.day:02d}'

        # Get observations per day for snow fall period
        opd_sm = snowCube.sel(time = slice(sms_str, sse_str)) # Filter to snow melt period
        opd_sm = opd_sm.notnull().sum(dim = 'time') / ((sse_date - sms_date).days + 1) # Count clear, divide by number of possible days
        opd_sm.rio.write_crs(crs, inplace = True) # Reapply crs
        clearObsPerDay['snowMelt_clearObsPerDay'] = opd_sm.astype('float32') # Add to clearObsPerDay     

        if verbose == True:
            print('Added snow melt period (' + sms_str + ' to ' + sse_str + ') to clearObsPerDay.') 

    # Returns xarray dataset with each array as a variable.
    return clearObsPerDay

######################################################################################################################################################
# IMS ################################################################################################################################################
######################################################################################################################################################

# Converts daily northern hemisphere IMS data (as downloaded) to annual Canada IMS xarray DataArray.
def createAnnualCanadaIMS(folder, wkt, canada, verbose = True):

    """
    Parameters:
    folder (str): Folder where downloaded IMS data are stored ('C:/path/to/folder').

    wkt (str): Text file where custom IMS WKT information is stored ('C:/path/to/ims_wkt2_2018.txt').

    canada (str): Folder where Canada shapefile is stored ('C:/path/to/file.shp'). Will be used for clipping and as projection in reproject. 

    verbose (bool): Whether (true) or not (false) to print function status  

    Returns:
    Annual Canada IMS cube (xarray DataArray)
    """

    # Open IMS data as a single cube with smallest dytpe
    ims = xr.open_mfdataset(folder + '/*.nc', # Downloaded as daily netCDF files
                            engine = 'netcdf4',  # Fastest
                            chunks = {'time': 1, 'x': -1, 'y': -1}, # Good chunking (purely by time-step)
                            parallel = True, 
                            drop_variables = 'projection').to_dataarray().squeeze().astype('uint8') # Don't need projection (will set from wkt)
    
    ims['time'] = ims['time'].dt.floor('1D') # Set HMS to 0s (beginning in Nov 2024, times were 1 am - messing up stuff down the line)
    
    if verbose == True:
        print('Opened daily IMS data as annual uint8 xarray DataArray (shape = ' + str(ims.shape)[1:-1] + ').')
    
    # Apply custom CRS from provided WKT file
    with open(wkt, 'r') as file:
        wkt = file.read()

    ims.rio.write_crs(wkt, inplace = True)
    ims.rio.write_nodata(0, inplace = True) # Set No Data to 0 (for later)

    if verbose == True:
        print('Applied custom CRS (' + wkt[9:26] + ') from provided WKT file.')

    # Subset to Canada bounding box
    ims = ims[:, 6900:11600, 9000:14500] # Approximate Canada bounding box (some buffer to make sure not edge pixels impacted)

    if verbose == True:
        print('Subset to approximate Canada bounding box (shape = ' + str(ims.shape)[1:-1] + ').')

    # To run smoothly with Dask on workstation, split into monthly cubes and reproject/clip
    if verbose == True:
        print('... Splitting into monthly cubes for smooth Dask processing ...')

    canada = gpd.read_file(canada, engine = 'pyogrio') # Bring in Canada shapefile for projection and clipping
    ims_list = [] # For input into xr.concat(), will append with months that have time-steps

    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'January', 1, verbose) # January
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'February', 2, verbose) # February
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'March', 3, verbose) # March 
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'April', 4, verbose) # April 
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'May', 5, verbose) # May 
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'June', 6, verbose) # June
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'July', 7, verbose) # July
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'August', 8, verbose) # August
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'September', 9, verbose) # September
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'October', 10, verbose) # October
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'November', 11, verbose) # November
    ims_list = monthlyIMSprocess(ims, canada, ims_list, 'December', 12, verbose) # December 

    # Combine back into annual IMS cube
    ims = xr.concat(ims_list, dim = 'time')  

    if verbose == True:
        print('Merged back into annual cube (shape = ' + str(ims.shape)[1:-1] + ').') 

    return ims  

######################################################################################################################################################

# Converts annual IMS 4-cat cubes from createAnnualCanadaIMS() to snow cubes where snow = 1, non-snow = 0. 
def annualIMS2SnowCube(cube_yr1, cube_yr2, yrs = 1, verbose = True):

    """
    Parameters:
    cube_yr1 (dask dataArray): IMS cube from createAnnualCanadaIMS(). Contains start of winter of interest. 
    - 0 = Outside Canada, 1 = Water, 2 = Land, 3 = Ice, 4 = Snow

    cube_yr2 (dask dataArray): IMS cube from createAnnualCanadaIMS(). Contains end of winter of interest.
    - 0 = Outside Canada, 1 = Water, 2 = Land, 3 = Ice, 4 = Snow  

    yrs (int): Length (in years) of IMS snow cube. Supports 1 and 2. 
    - 1: Uses optimal winter year for Canada (from national % snow coverage analysis). Each year ranges from August 21 - August 20. 
    - 2: Creates a full 2-year cube (January 1 to December 31). Beneficial for HLS to help find clear observations before snow fall. 

    verbose (bool): Whether (true) or not (false) to pridnt function status  

    Returns:
    Cube where snow = 1, non-snow = 0, unclear = NaN. 
    """

    if verbose == True:
        print('Initial IMS time-steps: Start (n = ' + str(len(cube_yr1)) + '), End (n = ' + str(len(cube_yr2)) + ').')

    if yrs == 1:
        cube_yr1 = cube_yr1.sel(time = slice(str(cube_yr1.time.dt.year[0].values) + '-08-21', str(cube_yr1.time.dt.year[0].values) + '-12-31'))
        cube_yr2 = cube_yr2.sel(time = slice(str(cube_yr2.time.dt.year[0].values) + '-01-01', str(cube_yr2.time.dt.year[0].values) + '-08-20'))

        if verbose == True:
            print('Filtered to winter year (Aug 21 - Aug 20): Start (n = ' + str(len(cube_yr1)) + '), End (n = ' + str(len(cube_yr2)) + ').')

    # Combine year 1 and year 2 cube along time dimension
    snowCube = xr.concat((cube_yr1, cube_yr2), dim = 'time').sortby('time') 

    if verbose == True:
        print('Combined into full cube (n = ' + str(len(snowCube)) + ').')

    # Visual check shows that there are no notable artifacts (e.g., 0s everywhere for one day) - No timestepClean required. 

    # Reclassify to snow = 1, non-snow ground observations = 0
    crs = snowCube.rio.crs # xarray where removes crs info... 

    snowCube = xr.where((snowCube == 1) | (snowCube == 2), 0, snowCube) # Non-snow (1, 2) = 0
    snowCube = xr.where((snowCube == 3) | (snowCube == 4), 1, snowCube) # Snow (3, 4) = 1
    snowCube.rio.write_crs(crs, inplace = True) # Reapply crs

    if verbose == True:
        print('Reclassifed (Snow = 1, Non-snow = 0)')

    return snowCube

######################################################################################################################################################

# Load, reproject, and clip annual IMS cube in monthly increments. 
def monthlyIMSprocess(ims, canada, ims_list, month, month_int, verbose = True):

    """
    Parameters:
    ims (xarray DataArray): Annual IMS cube for processing. 

    canada (gdf): Canada area geodataframe. 

    ims_list (list of xarray DataArray): List of processed monthly IMS cubes for later concatination. 

    month (str): Month to process. 

    month_int (int): Integer of month to process.

    verbose (bool): Whether (true) or not (false) to print function status 

    Returns:
    ims_list (list of xarray DataArray): Updated list of processed monthly IMS cubes for later concatination. 
    """      

    ims_month = ims.sel(time = ims['time.month'] == month_int) # All time-steps of certain month

    if len(ims_month) > 0: # If there are time-steps in this month

        if verbose == True:
            print('... ' + month + ' ...')   

        # Load into memory
        ims_month = pputil.loadXR(ims_month)

        if verbose == True:
            print('Loaded into memory (shape = ' + str(ims_month.shape)[1:-1] + ').')

        # Reproject to Canada CRS
        ims_month = ims_month.rio.reproject(dst_crs = canada.crs, resolution = 1000, resampling = Resampling.nearest) # 1 km is default resolution

        if verbose == True:
            print('Reprojected to ' + str(canada.crs) + ' (shape = ' + str(ims_month.shape)[1:-1] + ').')

        # Clip to Canada
        ims_month = ims_month.rio.clip(canada.geometry.values)

        if verbose == True:
            print('Clipped to Canada (shape = ' + str(ims_month.shape)[1:-1] + ').')  

        ims_list.append(ims_month)

    return ims_list               

######################################################################################################################################################
# General Snow Cube ##################################################################################################################################
######################################################################################################################################################

# Clean snow cubes by identifying clear snow periods and removing outliers. 
def cleanSnowCube(snowCube, form = 'binary', temporal = 'gaps', implausible_snow = [], thresh = 2, verbose = True):

    """
    Parameters:
    snowCube (xarray dataArray): xarray snow cube (e.g., output from annualFmask2SnowCube). 

    form (str): Type of snow cube to clean. "binary" = Snow vs. No-snow (e.g., Fmask). "percentage" = Snow % cover (not yet supported).

    temporal (str): Temporal status of snow cube to clean. 'gaps' = Temporal gaps in cube (e.g., HLS). 'continuous' = All days included (e.g., IMS).

    implausible_snow (list of two int): Julian days between which it is implausible for there to be snow cover. [] is default (don't apply check).
    - All observations between (>,<) these dates will be set to 0 (since we know that there is no snow, even cloudy observations can be confirmed)

    thresh (int): Number of consecutive observations of snow/non-snow required to maintain value in cleaned cube (otherwise set to NaN). 
    - This value x2 - 1 is used for an initial rolling median as well

    Returns:
    SnowCube with clear winter snow period (e.g., errors and outliers removed).

    verbose (bool): Whether (true) or not (false) to print function status  
    """

    crs = snowCube.rio.crs # xarray where removes crs info... 

    if form == 'binary':

        # Make a copy of the input snowCube, since is needed for portions of cumsum when there are NaNs in original snowCube
        if temporal == 'gaps': 
            snowCube_c = snowCube.copy()

        # Remove observations classified as snow during peirods where snow is implausible
        if len(implausible_snow) == 2: # Julian date range provided

            # Create day-of-year cube to use for identifying dates where it is implausible to be snow (no leap year fix, buffering anyway later)
            snowCube_doy = (snowCube['time.dayofyear'] > implausible_snow[0]) & (snowCube['time.dayofyear'] < implausible_snow[1])

            # For time-steps in-between implausible_snow dates, set all values to 0 (non-snow)
            snowCube_c = xr.where(snowCube_doy == 1, 0, snowCube_c)

            if verbose == True:
                yr2 = int(snowCube.time.dt.year[1]) # Last year represented in cube

                date1 = dt.datetime(yr2 - 1, 12, 31) + dt.timedelta(implausible_snow[0])
                start = f'{date1.month:02d}-{date1.day:02d}'

                date2 = dt.datetime(yr2 - 1, 12, 31) + dt.timedelta(implausible_snow[1])
                end =  f'{date2.month:02d}-{date2.day:02d}'

                print('Set all observations to non-snow between implausible snow dates (' + start + ' to ' + end + ').')

        # Calculate a rolling median, with window length = thresh x 2 - 1.
        if temporal == 'continuous':
            window = thresh * 2 - 1 # Align median window minium size with thresh
            snowCube_c = snowCube.rolling(time = window, min_periods = 1, center = True).construct('window').median('window').astype('uint8')
            # Convert to smallest dtype, this has side-effect of changing any 0.5s to 0 (where min_periods is even and matching). 
            # Makes sense to do this when edges are summer.
            # Construct version seems faster/more memory efficient than basic

            if verbose == True:
                print('Calculated rolling median through time of length ' + str(window) + '.')
        
        # Generate snow groups using cumulative sums
        # Create an xarray where cumsum is calculated but resets each time non-snow (0) is found
        cumsum = snowCube_c.cumsum(dim = 'time')
        cumsum_r = cumsum - cumsum.where(snowCube_c == 0).ffill(dim = 'time').fillna(0)

        if verbose == True:
            print('Calculated cumulative sum that resets when non-snow is found.')

        # Find groups that meet condition
        snow_grps = xr.full_like(cumsum_r, fill_value = 0) # Something to put results in
        snow_grps = xr.where(cumsum_r >= thresh, 1, snow_grps) # At least this many consecutive observation required
        snow_grps = xr.where((cumsum_r > 0) & (cumsum_r < thresh), np.nan, snow_grps) # Other valid sections over 0 becomes NaNs to be filled
        snow_grps = snow_grps.bfill(dim = 'time') #.fillna(1) # Backfill ones previously set to np.nan

        if verbose == True:
            print('Identified periods when ' + str(thresh) + '+ consecutive snow observations occured (skipping NaN).')

        # Generate non-snow groups using cumulative sums
        # Create flipped snowCube (Non-snow = 1, Snow = 0)
        snowCube_cf = xr.where(snowCube_c == 1, 0, snowCube_c)
        snowCube_cf = xr.where(snowCube_c == 0, 1, snowCube_cf)

        if verbose == True:
            print('Created flipped snowCube to identify non-snow periods.')

        # Create an xarray where cumsum is calculated but resets each time snow (0) is found
        cumsum = snowCube_cf.cumsum(dim = 'time')
        cumsum_r = cumsum - cumsum.where(snowCube_cf == 0).ffill(dim = 'time').fillna(0)

        if verbose == True:
            print('Calculated cumulative sum that resets when snow is found.')

        # Find groups that meet condition
        nosnow_grps = xr.full_like(cumsum_r, fill_value = 0) # Something to put results in
        nosnow_grps = xr.where(cumsum_r >= thresh, 1, nosnow_grps) # At least this many consecutive observation required
        nosnow_grps = xr.where((cumsum_r > 0) & (cumsum_r < thresh), np.nan, nosnow_grps) # Other valid sections over 0 becomes NaNs to be filled
        nosnow_grps = nosnow_grps.bfill(dim = 'time') #.fillna(1) # Backfill ones previously set to np.nan

        if verbose == True:
            print('Identified periods when ' + str(thresh) + '+ consecutive non-snow observations occured (skipping NaN).')

        # Use thresholded groups to clean
        snowCube_c = xr.where(snow_grps == 1, 1, np.nan).astype('float32') # >= thresh cumulative snow observations = 1
        snowCube_c = xr.where(nosnow_grps == 1, 0, snowCube_c) # >= thresh cumulative non-snow observations = 0
        if temporal == 'gaps':
            snowCube_c = xr.where(snowCube.isnull(), np.nan, snowCube_c) # Reset np.nans (otherwise will set snow/no-snow in uncertainty period)

        if verbose == True:
            print('Removed (NaN) snow and non-snow periods below ' + str(thresh) + ' consecutive observations.')

        # For continuous cube, interpolate_na here to fill in new transition gaps since dailySnowCube() is not required
        if temporal == 'continuous':
            snowCube_c = (snowCube_c.chunk({'time': -1, 'x': 'auto', 'y': 'auto'}) # Re-chunk spatially (req for interpolations)
                                    .interpolate_na(dim = 'time', method = 'nearest') # NN for binary 
                                    .astype('uint8')) # If 1 & 0 are odd distrance apart, middle date remains NA, this casts it to 0 with warning           
            
            if verbose == True:
                print('Filled in NaN transition days with nearest neighbor.')

        snowCube_c.rio.write_crs(crs, inplace = True) # Reapply crs

    return snowCube_c

######################################################################################################################################################

# Creates daily winter year snow cube where all pixels have values between 0 (Non-Snow) and 1 (Snow). With >0, <1 indicating uncertainty. 
def dailySnowCube(snowCube_clean, doy = '', verbose = True):

    """
    Parameters:
    snowCube_clean (xarray dataArray): SnowCube with clear winter snow period (e.g., errors and outliers removed).
    - Can be dask (faster), or in-memory

    doy (int): Day of year to split input cube for creating winter year daily snow cube. 
    - Starts cube day after doy and ends on doy next year. 
    - In Canada scenarios, should reasonably be during summer period (should be after Feb or may run into minor leap year issues)

    verbose (bool): Whether (true) or not (false) to print function status  

    Returns:
    snowCube_1D: Daily in-memory interpolated snow cube. 
    - Values between 0.01 and 0.99 can be considered uncertainty dates from binary input (i.e., dates in between snow and non-snow). 
    """

    # First, find values of NaN observations
    snowCube_1D = snowCube_clean.chunk({'time': -1, 'x': 'auto', 'y': 'auto'}).interpolate_na(dim = 'time', method = 'linear') 
    # NaNs become linear values of 1/0 (e.g., 1, NaN, 0 > 1, 0.5, 0)
    # Need to re-chunk spatially because interpolations require knowledge of all time-steps

    if verbose == True:
        print('Filled in all NaNs in snowCube with linearly interpolated values in time.')

    # From testing found it is better to load into memory before final interpolation
    #snowCube_1D = pputil.loadXR(snowCube_1D)

    # if verbose == True:
    #     print('Loaded filled snowCube into memory (faster than 1D interpolation with Dask from testing).')

    # Resample to daily, with remaining dates becoming linear values of 1/0
    snowCube_1D = snowCube_1D.resample(time = '1D').interpolate('linear').astype('float32') 
    # Defaults to float64

    if verbose == True:
        first_day = str(snowCube_1D['time'][0].values)[0:10]
        last_day = str(snowCube_1D['time'][-1].values)[0:10]
        print('Resampled to daily snowCube (' + first_day + ' to ' + last_day + ', n = ' + str(len(snowCube_1D)) + ') using linear interpolation.')

    # Filter and (if needed) Reindex
    yr2 = int(snowCube_1D.time.dt.year[-1]) # Last year represented in cube

    # Convert day-of-year to filterable strings
    if ((yr2 - 1) / 4).is_integer() == False: # Not a leap year
        date1 = dt.datetime(yr2 - 1, 1, 1) + dt.timedelta(doy) # Day after first year
    if ((yr2 - 1) / 4).is_integer() == True: # Leap year
        date1 = dt.datetime(yr2 - 1, 1, 1) + dt.timedelta(doy + 1) # Day after first year
    if (yr2 / 4).is_integer() == False: # Not a leap year (doy )
        date2 = dt.datetime(yr2, 1, 1) + dt.timedelta(doy - 1) # Day-of-year next year
    if (yr2 / 4).is_integer() == True: # Leap year
        date2 = dt.datetime(yr2, 1, 1) + dt.timedelta(doy) # Day-of-year next year

    start = f'{date1.year}-{date1.month:02d}-{date1.day:02d}'
    end =  f'{date2.year}-{date2.month:02d}-{date2.day:02d}'

    # Filter
    snowCube_1D = snowCube_1D.sel(time = slice(start, end))

    if verbose == True:
        first_day = str(snowCube_1D['time'][0].values)[0:10]
        last_day = str(snowCube_1D['time'][-1].values)[0:10]
        print('Filtered to winter year (' + first_day + ' to ' + last_day + ', n = ' + str(len(snowCube_1D)) + ').')

    time_index = pd.date_range(start = start, end = end) # All Days possible in cube

    if len(snowCube_1D) != len(time_index): # Only reinidex if snowCube_1D is missing days on the edges
        snowCube_1D = snowCube_1D.reindex({'time': time_index}, method = 'nearest') # Does not fill all NaNs because nearest neighbor may be NaN
        snowCube_1D = snowCube_1D.ffill(dim = 'time').bfill(dim = 'time') # Fill edge NaNs with closest 0/1

        if verbose == True:
            first_day = str(snowCube_1D['time'][0].values)[0:10]
            last_day = str(snowCube_1D['time'][-1].values)[0:10]
            print('Reindexed to full year cube (' + first_day + ' to ' + last_day + ', n = ' + str(len(snowCube_1D)) + ').')

    if len(snowCube_1D) == len(time_index):
        if verbose == True:
            print('Reinidexing not required since daily snowCube already contains all possible dates.')

    return snowCube_1D

######################################################################################################################################################

# Converts daily snow cube (e.g., from dailySnowCube()) to output Dataset quantifying various snow dynamics.
def dailySnowCube2SnowDynamics(snowCube_1D, products = ['start', 'end', 'length', 'periods', 'status'], uncertainty = True, verbose = True):

    """
    Parameters:
    snowCube_1D (xarray dataArray): Daily winter year snow cube. 

    products (list of str): Snow dynamics products of interest. Supports: ['start', 'end', 'length', 'periods']
    - 'start': Start day of first 'snow period', defined as number of days from December 31 in winter year. 
    - 'end': End day of the last 'snow period', defined as number of days from December 31 in winter year.
    - 'endSum': Number of days with snow cover since December 31 in winter year. 
    - 'length': Number of days with snow cover
    - 'periods': Number of 'snow periods', defined as number of periods of time where snow was observed (separated by non-snow)
    - 'status': Snow status. 0 = Regular snow fall/melt. 1 = Perennial snow cover. 2 = Inconsistent perennial snow cover. 3 = Snow free. 
    - 'pSnow_month': Percent of each month with snow cover. Needs to be run seperately from other products since it forms a monthly dataset. 
    - Note: 'status' requires 'length' to calculate. 

    uncertainty (bool): Whether (True) or not (False) to return uncertainty of eligible products
    - Eligible products: 'length', 'start', 'end'
    - Uncertainty is defined as the number of days between satellite snow and non-snow observation / 2 (i.e., +- when we are unsure of transition)
    - If Uncertainty is True, then either both or neither of 'start' and 'end' can be in products. Both are needed to find split date. 

    verbose (bool): Whether (true) or not (false) to print function status  

    Returns:
    snowDynamics: Snow dynamics produts (xarray Dataset). 
    """

    # Create empty xarray Dataset to fill with selected products
    crs = snowCube_1D.rio.crs # Some xarray functions remove crs 
    snowDynamics = xr.Dataset(coords = dict(x = ('x', snowCube_1D['x'].values), y = ('y', snowCube_1D['y'].values)))
    snowDynamics.rio.write_crs(crs, inplace = True) # Reapply crs

    if verbose == True:
        print('Created empty snowDynamics Dataset to fill.')    

    # If required, create binary snow cube, cutting linearly interpolated (uncertainty) values down the middle in time
    if np.issubdtype(snowCube_1D, np.floating) == True: # Only required for floating dtype cubes (int cubes won't need this)
        snowCube_b = xr.where(snowCube_1D >= 0.4999, 1, snowCube_1D) # If no uncertainty (e.g., IMS - don't need this)
        snowCube_b = xr.where(snowCube_1D < 0.4999, 0, snowCube_b) # 0.4999 accounts for weird rounding in some cases

        if verbose == True:
            print('Created binary snowCube (>= 0.5 = 1, < 0.5 = 0).')

    if np.issubdtype(snowCube_1D, np.integer) == True:
        snowCube_b = snowCube_1D.copy() 

    if ('start' in products) |  ('end' in products) | ('periods' in products):
        # Create cumsum of snowCube_b (usedby start, end, periods)
        b_cumsum = snowCube_b.cumsum(dim = 'time') # Turns to uint64 here

        # Also find number of days in the first year in cube (to calculate start/end from Dec 31 later)
        yr1_days = len(snowCube_b.sel(time = snowCube_b['time'][0].values.astype('datetime64[Y]').astype(str)))

        if verbose == True:
            print('Calculated cumulative sum of binary snowCube (used in multiple products).')

    # Create snow period start product (Days from Dec 31)
    if 'start' in products:
        # Find date where cumsum starts (+1 because 0 start), in days from winter year start
        snow_start = (xr.where(b_cumsum == 0, 999, b_cumsum).argmin(dim = 'time') + 1) #.astype('float32')
        snow_start = snow_start - yr1_days # Recalculate to days from Dec 31
        # Remove any where first day in winter year is snow (perennial), no snow already gets captured by this and set to NaN
        snow_start = xr.where(snow_start == 1 - yr1_days, np.nan, snow_start) 
        snow_start.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics['snow_start'] = snow_start.astype('float32') # Add to snowDynamics

        if verbose == True:
            print('Added snow cover start date (# days from Dec 31) to snowDynamics.')

    # Create snow period end product (Days from Dec 31)
    if 'end' in products:
        snow_end = (b_cumsum.argmax(dim = 'time') + 1) # Index starts at 0
        snow_end = snow_end - yr1_days # Recalculate to days from Dec 31
        # Remove any where last day in winter year is snow (perennial) or when no snow is present (snow free)
        snow_end = xr.where((snow_end == len(b_cumsum) - yr1_days) | (snow_end == 1 - yr1_days), np.nan, snow_end) 
        snow_end.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics['snow_end'] = snow_end.astype('float32') # Add to snowDynamics   

        if verbose == True:
            print('Added snow cover end date (# days from Dec 31) to snowDynamics.') 

    # Create snow period end sum product (# Days)
    if 'endSum' in products:
        yr2 = str(int(snowCube_b.time.dt.year[-1]))
        snow_endSum = snowCube_b.sel(time = slice(yr2 + '-01-01', None)).sum(dim = 'time') # Sum of days with snow cover since Dec 31
        # Remove any where last day in winter year is snow (perennial) or when no snow is present (snow free)
        snow_endSum = xr.where((snow_endSum == len(b_cumsum) - yr1_days) | (snow_endSum == 1 - yr1_days), np.nan, snow_endSum)         
        snow_endSum.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics['snow_endSum'] = snow_endSum.astype('float32') # Add to snowDynamics

        if verbose == True:
            print('Added snow cover end date sum (# days from Dec 31 with snow cover) to snowDynamics.') 

    # Create snow cover length product (# Days)
    if 'length' in products:
        snow_length = snowCube_b.sum(dim = 'time') # Sum of days with snow cover
        snow_length.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics['snow_length'] = snow_length.astype('uint16') # Add to snowDynamics

        if verbose == True:
            print('Added snow cover length (# days with snow cover) to snowDynamics.') 

    # Create number of snow periods product (# Periods)
    if 'periods' in products:
         # Where the resetting cumsum increases past 0, set to 1, then sum to get number of times cumsum increases past 0
        cumsum = b_cumsum - b_cumsum.where(snowCube_1D == 0).ffill(dim = 'time').fillna(0) # snowCube_b?
        snow_periods = xr.full_like(cumsum, fill_value = 0)
        snow_periods = xr.where(cumsum == 1, 1, snow_periods).sum(dim = 'time')
        snow_periods.rio.write_crs(crs, inplace = True) # Reapply crs

        snowDynamics['snow_periods'] = snow_periods.astype('uint8') # Add to snowDynamics 

        if verbose == True:
            print('Added snow period count (# seperated snow periods) to snowDynamics.') 

    # Create snow status product (0 = Regular Fall/Melt, 1 = Perennial, 2  = Inconsistent Perennial, 3 = Snow Free)
    if 'status' in products:
        snow_status = xr.where((snowCube_b[0] == 1) | (snowCube_b[-1] == 1), 2, 0) # 2 = Inconsistent Perennial (first/last date)
        #snow_status = (snowDynamics['snow_length'] >= len(snowCube_b)) # 1 = Perennial, 0 = Regular Snow Fall/Melt
        snow_status = xr.where(snowDynamics['snow_length'] >= len(snowCube_b), 1, snow_status)
        snow_status = xr.where(snowDynamics['snow_length'] == 0, 3, snow_status)
        snow_status.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics['snow_status'] = snow_status.astype('uint8') # Add to snowDynamics 

        if verbose == True:
            print('Added snow status (0 = Regular fall/melt, 1 = Perennial, 2 = Inconsistent perennial, 3 = Snow free) to snowDynamics.')

    if 'pSnow_month' in products:
        pSnow = snowCube_b.groupby('time.month').sum('time') # Number of snow days per month
        month_len = snowCube_b.time.dt.days_in_month.groupby('time.month').mean() # Number of days per month
        pSnow = ((pSnow / month_len) * 100).round() # Normalized to % snow per month (0 - 100 ints)
        pSnow.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics['pSnow'] = pSnow.astype('uint8') # Add to snowDynamics

        if verbose == True:
            print('Added percent snow coverage for each month to snowDynamics.')

    # Create uncertainty versions of eligible products
    if uncertainty == True:

        # Create uncertainty snow cube, isolating dates in between satellite observed snow and non-snow (> 0.01 and < 0.99 = Uncertain)
        # Use these values since interpolation can sightly change observed values from 0 and 1
        snowCube_u = xr.where((snowCube_1D < 0.005) | (snowCube_1D > 0.995), 0, snowCube_1D)
        snowCube_u = xr.where((snowCube_1D >= 0.005) & (snowCube_1D <= 0.995) , 1, snowCube_u) 
        snowCube_u.rio.write_crs(crs, inplace = True) # Reapply crs

        if verbose == True:
            print('Created uncertainty snowCube (>= 0.005 & <= 0.995 = 1, else = 0).')

        # Find gap between snow_start and snow_end for local uncertainty metrics around those times
        if ('start' in products) | ('end' in products):
            # This method overestimates start or end uncertainty if there are multiple snow periods (internal uncertanties incuded in sum)
            # Requires computing snow_start and snow_end (~2+ mins)

            snow_start_inmem = snow_start.compute() # Needed to avoid double load if quantiles are required... 
            snow_end_inmem = snow_end.compute() # Needed to avoid double load if quantiles are required... 
            start_last = int(snow_start_inmem.max()) # Latest snow start date in area
            end_first = int(snow_end_inmem.min()) # Earliest snow end date in area
            endstart_diff = end_first - start_last # Difference between

            if endstart_diff < 0: # If snow_start and snow_end overlap, check 99% start vs 1% end quantiles
                start_last = int(snow_start_inmem.quantile(0.99)) # Latest* snow start date in area
                end_first = int(snow_end_inmem.quantile(0.01)) # Earliest* snow end date in area
                endstart_diff = end_first - start_last # Difference between

            if endstart_diff < 0: # If snow_start and snow_end still overlap, check 98% start vs 2% end quantiles
                start_last = int(snow_start_inmem.quantile(0.98))
                end_first = int(snow_end_inmem.quantile(0.02))
                endstart_diff = end_first - start_last

            if endstart_diff >= 0: # If regional snow_start and snow_end do not overlap

                # Convert latest start and earliest end to dates, and find split date between
                start_last = dt.datetime(int(snowCube_1D.time.dt.year[-1] - 1), 12, 31) + dt.timedelta(start_last)
                end_first = dt.datetime(int(snowCube_1D.time.dt.year[-1] - 1), 12, 31) + dt.timedelta(end_first)
                split_date = start_last + (end_first - start_last) / 2 # Split date
                split_date_1 = split_date + dt.timedelta(1) # Day after split date for filtering
                split_date =  f'{split_date.year}-{split_date.month:02d}-{split_date.day:02d}' # String for filtering

                if verbose == True:
                    start_last = f'{start_last.year}-{start_last.month:02d}-{start_last.day:02d}' # Printable string
                    end_first = f'{end_first.year}-{end_first.month:02d}-{end_first.day:02d}' # Printable string  

                    print('Used ' + str(endstart_diff) + ' day gap between latest snow start (' + start_last + ') and earliest snow end (' + 
                          end_first + ') to find winter year split day: ' + split_date + '.')

            if endstart_diff < 0: # If snow_start and snow_end still overlap... update this function as Exception says. 
                raise Exception('Snow start and end still overlap in time, update uncertainty procedure.') 
        
        # Start
        if 'start' in products:
            snow_start_u = snowCube_u.sel(time = slice(None, split_date)).sum(dim = 'time') / 2 # +- number of uncertain days between 0/1
            snow_start_u = xr.where(snow_start.notnull(), snow_start_u, np.nan) # Align NaNs with snow_start
            snow_start_u.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics['snow_start_u'] = snow_start_u.astype('float32') # Add to snowDynamics 

            if verbose == True:
                print('Added snow start uncertainty (# days between last non-snow and first snow observation / 2) to snowDynamics.')
    
        # End
        if 'end' in products: 
            split_date_1 = f'{split_date_1.year}-{split_date_1.month:02d}-{split_date_1.day:02d}' # String for filtering
            snow_end_u = snowCube_u.sel(time = slice(split_date_1, None)).sum(dim = 'time') / 2 # +- number of uncertain days between 0/1
            snow_end_u = xr.where(snow_end.notnull(), snow_end_u, np.nan) # Align NaNs with snow_end
            snow_end_u.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics['snow_end_u'] = snow_end_u.astype('float32') # Add to snowDynamics 

            if verbose == True:
                print('Added snow end uncertainty (# days between last snow and first non-snow observation / 2) to snowDynamics.')

        # Length
        if 'length' in products:
            snow_length_u = snowCube_u.sum(dim = 'time') / 2 # +- number of uncertain days between 0/1
            snow_length_u.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics['snow_length_u'] = snow_length_u.astype('float32') # Add to snowDynamics 

            if verbose == True:
                print('Added snow length uncertainty (total # days between snow and non-snow observations / 2) to snowDynamics.')

    # Returns xarray dataset with each array as a variable.
    return snowDynamics

######################################################################################################################################################

# From dailySnowCube2SnowDynamics() output cube, create inter-annual (multi-year) snow dynamics products.
def interannualSnowDynamics(snowDynamics, min_count = 'half', products = ['start', 'end', 'length', 'periods', 'status'], 
                            uncertainty = ['start_u', 'end_u', 'length_u'], clearObsPerDay = ['year', 'season', 'fall', 'peak', 'melt'], 
                            form = 'mean_weighted', implausible_snow = [], sd = False, quality = True, best_value = True, verbose = True):

    """
    Parameters:
    snowDynamics (xarray dataSet): winterYear snow dynamics product produced from dailySnowCube2SnowDynamics()
    - It is expected that snowDynamics is in memory

    min_count (str): How to handle NaNs for each pixel, applies to all products. Supports: 'all', 'half', 'one'.
    - 'all': All winterYears should have a value.
    - 'half': At least half of winterYears should have a value (e.g., At least 3 values over 5 winterYears).
    - 'one': At least one winterYear should have a value.
    - Start and end will have additional NaNs compared to other products in perennial/snow free cases. 

    products (list of str): Snow dynamics products of interest. Supports: ['start', 'end', 'endSum', 'length', 'periods', 'status'].
    - 'start': Start day of first 'snow period'. Impacted by form. 
    - 'end': End day of the last 'snow period'. Impacted by form. 
    - 'endSum': Number of days with snow cover since December 31 in winter year. Impacted by form. 
    - 'length': Number of days with snow cover. Impacted by form. 
    - 'periods': Number of 'snow periods'. Impacted by form. 
    - 'status': Snow status. Interannual: % Years with perennial snow, % Years snow free. 

    uncertainty (list of str): Snow dynamics uncertainty products of interest. Supports: ['start_u', 'end_u', 'length_u'].
    - 'start_u': Uncertainty in start day of first 'snow period'. Impacted by form. 
    - 'end_u': Uncertainty in end day of last 'snow period'. Impacted by form. 
    - 'length_u': Total uncertainty across snow season. Impacted by form. 

    clearObsPerDay (list of str): Snow season clear observation products of interest. Supports: ['year', 'season', 'fall', 'peak', 'melt'].
    - 'year': Clear observations per day over winter year. Implacted by form. 
    - 'season': Clear observations per day over snow season . Implacted by form.  
    - 'fall': Clear observations per day over snow fall period. Implacted by form. 
    - 'peak': Clear observations per day over snow peak period. Implacted by form. 
    - 'melt': Clear observations per day over snow melt period. Implacted by form. 

    form (str): Type of interannial averaging to do. Supports: 'mean', 'mean_clean', 'mean_weighted'
    - 'mean': Regular mean. Applies to start, end, length, priods, start_u, end_u, length_u, year, season, fall, peak, melt. 
    - 'mean_clean': 'Regular mean, removing observations with higher (> 15) uncertainty. Requires uncertainty products. 
    - 'mean_weighted': Mean weighted by uncertainty (higher uncertainty = lower weight) and plausability (more implausible = lower weight). 
    - mean_clean requires uncertainty products. mean_weighted requires uncertainty products and IMS stats.
    - mean_clean and mean_weighted applies to start, end, length, start_u, end_u, length_u. 

    implausible_snow (list of two int): Julian days between which it is implausible for there to be snow cover. [] is default (don't apply check).
    - Only applies to 'mean_weighted'
    - If no implausible snow provided, implausibility weight will be based purely on difference from median. 

    sd (bool): Whether (True) or not (False) to return interannual standard deviation products in addition to mean
    - Eligible for start, end, length, periods

    quality (bool): Whether (True) or not (False) to return interannual mean quality products. Only applies if form = 'mean_weighted'.
    - This is essentially the weighted mean of the weights by mean_weighted. Higher values = lower uncertainty and higher plausibility. 
    - Eligible for start, end, length. 

    best_value (bool): Whether (True) or not (False) to return interannual best value products. Only applies if form = 'mean_weighted'. 
    - For elgiible products, will return the observation with the highest quality (weight) and the year of that observation. 
    - Will also return the highest quality value if quality == True. 
    - Eligible for start, end, length and their uncertainties.

    verbose (bool): Whether (true) or not (false) to print function status.  

    Returns:
    snowDynamics: Snow dynamics produts (xarray Dataset). 
    """

    # Calculate min_count
    if min_count == 'all':
        min_count = snowDynamics.sizes['winterYear']
    if min_count == 'half':
        min_count = math.ceil(snowDynamics.sizes['winterYear'] / 2) # Half, rounded up to nearest int
    if min_count == 'one':
        min_count = 1

    # Create empty xarray Dataset to fill with selected products
    crs = snowDynamics.rio.crs # Some xarray functions remove crs 
    snowDynamics_i = xr.Dataset(coords = dict(x = ('x', snowDynamics['x'].values), y = ('y', snowDynamics['y'].values)))
    snowDynamics_i.rio.write_crs(crs, inplace = True) # Reapply crs

    # Create snow period start interannual product
    if 'start' in products:
        
        if form == 'mean_clean': # Only observations wth <= 15 uncertainty
            snowDynamics['snow_start'] = xr.where(snowDynamics['snow_start_u'] <= 15, snowDynamics['snow_start'], np.nan) 

        valid_count = snowDynamics['snow_start'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_start'] = xr.where(valid_count >= min_count, snowDynamics['snow_start'], np.nan) # Filter by valid_count

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)     
            snow_start_mn = snowDynamics['snow_start'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count)

            # Uncertainty (50%)
            weights_u = 2.718 ** (-0.046 * snowDynamics['snow_start_u']) # -0.046 means 50% weight on 15, exponential

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get doys
                early_start = implausible_snow[1] - 365 # Before this date = early start
                #late_start = int(snowDynamics['snow_start'].chunk({'winterYear': -1, 'x': -1, 'y': -1}).quantile(0.95))  # After this = late start 
                late_start = int(snowDynamics['snow_start'].quantile(0.95))  # After this = late start 

                # Get diffs
                early_diff = abs(snowDynamics['snow_start'] - early_start) # Negative (before abs) = Before early start
                late_diff = snowDynamics['snow_start'] - late_start # Positive = After late start

                # Isolate outside
                outside_diff = xr.where(snowDynamics['snow_start'] >= early_start, 0, early_diff) # Add early end days
                outside_diff = xr.where(snowDynamics['snow_start'] <= late_start, outside_diff, late_diff) # Add late end days

                # Get weights
                weights_i1 = 2.718 ** (-0.046 * outside_diff) # -0.046 means 50% weight on 15, exponential

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diff = abs(snowDynamics['snow_start'] - snowDynamics['snow_start'].median(dim = 'winterYear', skipna = True))
            weights_i2 = 2.718 ** (-0.046 * med_diff) # -0.046 mean 50% weight on 15, exponential    

            # Combine weights (weighted sum) and calculate weighted mean
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_start_c = (weights_u * 0.5) + (weights_i1 * 0.25) + (weights_i2 * 0.25) # 50% weight for uncertainty, 50% for implausibility
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_start_c = (weights_u * 0.5) + (0.25) + (weights_i2 * 0.25) # No implausible dates, 0.25 weight by default
            snow_start_mn = snowDynamics['snow_start'].weighted(weights_start_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_start_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_start_mn'] = snow_start_mn # Add to snowDynamics

        if sd == True:
            snow_start_sd = snowDynamics['snow_start'].std(dim = 'winterYear', skipna = True)
            snow_start_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_start_sd'] = snow_start_sd # Add to snowDynamics

        if verbose == True:
            print('Added snow cover start date to interannual snowDynamics.')

    # Create snow period end interannual product
    if 'end' in products:

        if form == 'mean_clean': # Only observations wth <= 15 uncertainty
            snowDynamics['snow_end'] = xr.where(snowDynamics['snow_end_u'] <= 15, snowDynamics['snow_end'], np.nan) 

        valid_count = snowDynamics['snow_end'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_end'] = xr.where(valid_count >= min_count, snowDynamics['snow_end'], np.nan) # Filter by valid_count

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)   
            snow_end_mn = snowDynamics['snow_end'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count)

            # Uncertainty (50%)
            weights_u = 2.718 ** (-0.046 * snowDynamics['snow_end_u']) # -0.046 means 50% weight on 15, exponential

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get doys
                late_end = implausible_snow[0] # After this date = late end
                #early_end = int(snowDynamics['snow_end'].chunk({'winterYear': -1, 'x': -1, 'y': -1}).quantile(0.05)) # Before this = early end
                early_end = int(snowDynamics['snow_end'].quantile(0.05)) # Before this = early end

                # Get diffs
                late_diff = snowDynamics['snow_end'] - late_end # Positive = After late end
                early_diff = abs(snowDynamics['snow_end'] - early_end) # Negative (before abs) = Before early end 

                # Isolate outside 
                outside_diff = xr.where(snowDynamics['snow_end'] >= early_end, 0, early_diff) # Add early end days
                outside_diff = xr.where(snowDynamics['snow_end'] <= late_end, outside_diff, late_diff) # Add late end days

                # Get weights
                weights_i1 = 2.718 ** (-0.046 * outside_diff) # -0.046 means 50% weight on 15, exponential

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diff = abs(snowDynamics['snow_end'] - snowDynamics['snow_end'].median(dim = 'winterYear', skipna = True))  
            weights_i2 = 2.718 ** (-0.046 * med_diff) # -0.046 mean 50% weight on 15, exponential              

            # Combine weights (weighted sum) and calculate weighted mean
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_end_c = (weights_u * 0.5) + (weights_i1 * 0.25) + (weights_i2 * 0.25) # 50% weight for uncertainty, 50% for implausibility
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_end_c = (weights_u * 0.5) + (0.25) + (weights_i2 * 0.25) # No implausible dates, 0.25 weight by default        
            snow_end_mn = snowDynamics['snow_end'].weighted(weights_end_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_end_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_end_mn'] = snow_end_mn # Add to snowDynamics

        if sd == True:
            snow_end_sd = snowDynamics['snow_end'].std(dim = 'winterYear', skipna = True)
            snow_end_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_end_sd'] = snow_end_sd # Add to snowDynamics        

        if verbose == True:
            print('Added snow cover end date to interannual snowDynamics.') 

    # Create snow period end interannual product
    if 'endSum' in products:

        if form == 'mean_clean': # Only observations wth <= 15 uncertainty
            snowDynamics['snow_endSum'] = xr.where(snowDynamics['snow_end_u'] <= 15, snowDynamics['snow_endSum'], np.nan) 

        valid_count = snowDynamics['snow_endSum'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_endSum'] = xr.where(valid_count >= min_count, snowDynamics['snow_endSum'], np.nan) # Filter by valid_count

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)   
            snow_endSum_mn = snowDynamics['snow_endSum'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count)

            # Uncertainty (50%)
            weights_u = 2.718 ** (-0.046 * snowDynamics['snow_end_u']) # -0.046 means 50% weight on 15, exponential

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get doys
                late_end = implausible_snow[0] # After this date = late end
                #early_end = int(snowDynamics['snow_endSum'].chunk({'winterYear': -1, 'x': -1, 'y': -1}).quantile(0.05)) # Before this = early end
                early_end = int(snowDynamics['snow_endSum'].quantile(0.05)) # Before this = early end

                # Get diffs
                late_diff = snowDynamics['snow_endSum'] - late_end # Positive = After late end
                early_diff = abs(snowDynamics['snow_endSum'] - early_end) # Negative (before abs) = Before early end 

                # Isolate outside 
                outside_diff = xr.where(snowDynamics['snow_endSum'] >= early_end, 0, early_diff) # Add early end days
                outside_diff = xr.where(snowDynamics['snow_endSum'] <= late_end, outside_diff, late_diff) # Add late end days

                # Get weights
                weights_i1 = 2.718 ** (-0.046 * outside_diff) # -0.046 means 50% weight on 15, exponential

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diff = abs(snowDynamics['snow_endSum'] - snowDynamics['snow_endSum'].median(dim = 'winterYear', skipna = True))  
            weights_i2 = 2.718 ** (-0.046 * med_diff) # -0.046 mean 50% weight on 15, exponential              

            # Combine weights (weighted sum) and calculate weighted mean
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_endSum_c = (weights_u * 0.5) + (weights_i1 * 0.25) + (weights_i2 * 0.25) # 50% weight for uncertainty, 50% for implausibility
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_endSum_c = (weights_u * 0.5) + (0.25) + (weights_i2 * 0.25) # No implausible dates, 0.25 weight by default        
            snow_endSum_mn = snowDynamics['snow_endSum'].weighted(weights_endSum_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_endSum_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endSum_mn'] = snow_endSum_mn # Add to snowDynamics

        if sd == True:
            snow_endSum_sd = snowDynamics['snow_endSum'].std(dim = 'winterYear', skipna = True)
            snow_endSum_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_endSum_sd'] = snow_endSum_sd # Add to snowDynamics        

        if verbose == True:
            print('Added snow cover days since December 31 to interannual snowDynamics.') 

    # Create snow cover length interannual product
    if 'length' in products:
        snowDynamics['snow_length'] = snowDynamics['snow_length'].astype('float32') # Helps with math, may have NAs

        if form == 'mean_clean': # Only observations wth <= 30 uncertainty (15 + 15)
            snowDynamics['snow_length'] = xr.where(snowDynamics['snow_length_u'] <= 30, snowDynamics['snow_length'], np.nan)   

        valid_count = snowDynamics['snow_length'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_length'] = xr.where(valid_count >= min_count, snowDynamics['snow_length'], np.nan) # Filter by valid_count      

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)   
            snow_length_mn = snowDynamics['snow_length'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count)
            
            # Uncertainty (50%)
            weights_u = 2.718 ** (-0.023 * snowDynamics['snow_length_u']) # -0.023 means 50% weight on 30, exponential

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get lengths
                too_long = 365 - (implausible_snow[1] - implausible_snow[0]) # More than this many days = too much snow cover
                #too_short = int(snowDynamics['snow_length'].chunk({'winterYear': -1, 'x': -1, 'y': -1}).quantile(0.05)) # Less = too little
                too_short = int(snowDynamics['snow_length'].quantile(0.05)) # Less = too little

                # Get diffs
                long_diff = snowDynamics['snow_length'] - too_long # Positive = Too long
                short_diff = abs(snowDynamics['snow_length'] - too_short) # Negative (before abs) = Too short

                # Isolate outside 
                outside_diff = xr.where(snowDynamics['snow_length'] <= too_long, 0, long_diff) # Add too long snow cover
                outside_diff = xr.where(snowDynamics['snow_length'] >= too_short, outside_diff, short_diff) # Add too short snow cover

                # Get weights
                weights_i1 = 2.718 ** (-0.046 * outside_diff) # -0.046 means 50% weight on 15, exponential

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diff = abs(snowDynamics['snow_length'] - snowDynamics['snow_length'].median(dim = 'winterYear', skipna = True))  
            weights_i2 = 2.718 ** (-0.046 * med_diff) # -0.046 mean 50% weight on 15, exponential    

            # Combine weights (weighted sum) and calculate weighted mean   
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_length_c = (weights_u * 0.5) + (weights_i1 * 0.25) + (weights_i2 * 0.25) # 50% weight for uncertainty, 50% for implausibility
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_length_c = (weights_u * 0.5) + (0.25) + (weights_i2 * 0.25) # No implausible dates, 0.25 weight by default                     
            snow_length_mn = snowDynamics['snow_length'].weighted(weights_length_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_length_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_length_mn'] = snow_length_mn # Add to snowDynamics

        if sd == True:
            snow_length_sd = snowDynamics['snow_length'].std(dim = 'winterYear', skipna = True)
            snow_length_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_length_sd'] = snow_length_sd # Add to snowDynamics              

        if verbose == True:
            print('Added snow cover length to interannual snowDynamics.') 

    # Create number of snow periods interannual product (always regular mean)
    if 'periods' in products:
  
        valid_count = snowDynamics['snow_periods'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_periods'] = xr.where(valid_count >= min_count, snowDynamics['snow_periods'], np.nan) # Filter by valid_count
        snow_periods_mn = snowDynamics['snow_periods'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        snow_periods_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_periods_mn'] = snow_periods_mn.astype('float32') # Add to snowDynamics

        if sd == True:
            snow_periods_sd = snowDynamics['snow_periods'].std(dim = 'winterYear', skipna = True)
            #snow_periods_sd = xr.where(snow_periods_mn.isnull(), np.nan, snow_periods_sd) # Set NaN if mn NaN (based on min_count)
            snow_periods_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_periods_sd'] = snow_periods_sd.astype('float32') # Add to snowDynamics, default is float64              

        if verbose == True:
            print('Added snow period count to interannual snowDynamics.') 

    # Create snow status interannual product (0 = Regular Fall/Melt, 1 = Perennial, 2  = Inconsistent Perennial, 3 = Snow Free)
    if 'status' in products: # Note: status not impacted by mean rules     
        valid_count = snowDynamics['snow_status'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel

        # % years with perennial snow  
        pPerennialSnow = xr.where(snowDynamics['snow_status'] == 2, 1, snowDynamics['snow_status']) # 2 becomes 1 (1 already 1)
        pPerennialSnow = xr.where(pPerennialSnow == 3, 0, pPerennialSnow) # 3 becomes 0
        pPerennialSnow = (pPerennialSnow.sum(dim = 'winterYear', min_count = min_count) # Sum (pixels reaching min_count)
                          / valid_count * 100) # Divided by # winterYears with value * 100
        pPerennialSnow.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['pPerennialSnow'] = pPerennialSnow.astype('float32') # Add to snowDynamics, default is float64

        # % years snow-free
        pSnowFree = xr.where((snowDynamics['snow_status'] == 2) | ((snowDynamics['snow_status'] == 1)), 0, snowDynamics['snow_status']) # 2/1 become 0
        pSnowFree = xr.where(pSnowFree == 3, 1, pSnowFree) # 3 becomes 1
        pSnowFree = (pSnowFree.sum(dim = 'winterYear', min_count = min_count) # Sum (pixels reaching min_count)
                     / valid_count * 100) # Divided by # winterYears with value * 100  
        pSnowFree.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['pSnowFree'] = pSnowFree.astype('float32')  # Add to snowDynamics, default is float64                 

        if verbose == True:
            print('Added % years snow-free and with perennial snow to snowDynamics.')

    # Start: uncertainty, quality, best value and year
    # Uncertainty (requires start)
    if ('start' in products) & ('start_u' in uncertainty):

        snowDynamics['snow_start_u'] = xr.where(snowDynamics['snow_start'].notnull(), snowDynamics['snow_start_u'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count), using NaNs from snow_start to clean
            snow_start_u_mn = snowDynamics['snow_start_u'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count), can use weights from above
            snow_start_u_mn = snowDynamics['snow_start_u'].weighted(weights_start_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_start_u_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_start_u_mn'] = snow_start_u_mn # Add to snowDynamics  

        if verbose == True:
            print('Added snow cover start date uncertainty to interannual snowDynamics.') 

    # Quality (requires weighted mean and start)
    if (quality == True) & (form == 'mean_weighted') & ('start' in products): 
        snow_start_q_mn = weights_start_c.weighted(weights_start_c.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_start_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_start_q_mn'] = snow_start_q_mn      

        if verbose == True:
            print('Added snow cover start date quality to interannual snowDynamics.')

    # Best value (requires weighted mean and start)
    if (best_value == True) & (form == 'mean_weighted') & ('start' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_start_bvi = weights_start_c.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        #snow_start_bvi = weights_start_c.fillna(0).argmax(dim = 'winterYear', skipna = True).compute()

        # Get best value from best year
        snow_start_bv = snowDynamics['snow_start'].isel(winterYear = snow_start_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_start_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_start_bv'] = snow_start_bv

        if verbose == True:
            print('Added snow cover start date best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_start_bvy = xr.where(snow_start_bv.notnull(), snow_start_bvi + 2019, np.nan).astype('float32')
        snow_start_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_start_bvy'] = snow_start_bvy

        if verbose == True:
            print('Added snow cover start date best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_start_bvq = weights_start_c.isel(winterYear = snow_start_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_start_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_start_bvq'] = snow_start_bvq

            if verbose == True:
                print('Added snow cover start date best value quality to interannual snowDynamics')

    # End: uncertainty, quality, best value and year
    # Uncertainty (requires end or endSum)
    if (('end' in products) | ('endSum' in products)) & ('end_u' in uncertainty):

        snowDynamics['snow_end_u'] = xr.where(snowDynamics['snow_end'].notnull(), snowDynamics['snow_end_u'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count), using NaNs from snow_end to clean
            snow_end_u_mn = snowDynamics['snow_end_u'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count), can use weights from above
            snow_end_u_mn = snowDynamics['snow_end_u'].weighted(weights_end_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_end_u_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_end_u_mn'] = snow_end_u_mn # Add to snowDynamics  

        if verbose == True:
            print('Added snow cover end date uncertainty to interannual snowDynamics.') 

    # Quality (requires weighted mean and end)
    if (quality == True) & (form == 'mean_weighted') & ('end' in products): 
        snow_end_q_mn = weights_end_c.weighted(weights_end_c.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_end_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_end_q_mn'] = snow_end_q_mn      

        if verbose == True:
            print('Added snow cover end date quality to interannual snowDynamics.')   

    # Best value (requires weighted mean and end)
    if (best_value == True) & (form == 'mean_weighted') & ('end' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_end_bvi = weights_end_c.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        #snow_end_bvi = weights_end_c.fillna(0).argmax(dim = 'winterYear', skipna = True).compute()

        # Get best value from best year
        snow_end_bv = snowDynamics['snow_end'].isel(winterYear = snow_end_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_end_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_end_bv'] = snow_end_bv

        if verbose == True:
            print('Added snow cover end date best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_end_bvy = xr.where(snow_end_bv.notnull(), snow_end_bvi + 2019, np.nan).astype('float32')
        snow_end_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_end_bvy'] = snow_end_bvy

        if verbose == True:
            print('Added snow cover end date best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_end_bvq = weights_end_c.isel(winterYear = snow_end_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_end_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_end_bvq'] = snow_end_bvq

            if verbose == True:
                print('Added snow cover end date best value quality to interannual snowDynamics')

    # EndSum: quality, best value and year (uncertainty uses end_u)
    # Quality (requires weighted mean and end)
    if (quality == True) & (form == 'mean_weighted') & ('endSum' in products): 
        snow_endSum_q_mn = weights_endSum_c.weighted(weights_endSum_c.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_endSum_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endSum_q_mn'] = snow_endSum_q_mn      

        if verbose == True:
            print('Added snow cover days since December 31 quality to interannual snowDynamics.')   

    # Best value (requires weighted mean and end)
    if (best_value == True) & (form == 'mean_weighted') & ('endSum' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_endSum_bvi = weights_endSum_c.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        #snow_endSum_bvi = weights_endSum_c.fillna(0).argmax(dim = 'winterYear', skipna = True).compute()

        # Get best value from best year
        snow_endSum_bv = snowDynamics['snow_endSum'].isel(winterYear = snow_endSum_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_endSum_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endSum_bv'] = snow_endSum_bv

        if verbose == True:
            print('Added snow cover days since December 31 best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_endSum_bvy = xr.where(snow_endSum_bv.notnull(), snow_endSum_bvi + 2019, np.nan).astype('float32')
        snow_endSum_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endSum_bvy'] = snow_endSum_bvy

        if verbose == True:
            print('Added snow cover days since December 31 best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_endSum_bvq = weights_endSum_c.isel(winterYear = snow_endSum_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_endSum_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_endSum_bvq'] = snow_endSum_bvq

            if verbose == True:
                print('Added snow cover days since December 31 best value quality to interannual snowDynamics')

    # Length: uncertainty, quality, best value and year
    # Uncertainty (requires length)
    if ('length' in products) &  ('length_u' in uncertainty): 

        snowDynamics['snow_length_u'] = xr.where(snowDynamics['snow_length'].notnull(), snowDynamics['snow_length_u'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count), using NaNs from snow_length to clean
            snow_length_u_mn = snowDynamics['snow_length_u'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        if form == 'mean_weighted': # Weighted mean (with min_count), can use weights from above
            snow_length_u_mn = snowDynamics['snow_length_u'].weighted(weights_length_c.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_length_u_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_length_u_mn'] = snow_length_u_mn.astype('float32') # Add to snowDynamics  

        if verbose == True:
            print('Added snow season length uncertainty to interannual snowDynamics.')   

    # Quality (requires weighted mean and length)
    if (quality == True) & (form == 'mean_weighted') & ('length' in products): 
        snow_length_q_mn = weights_length_c.weighted(weights_length_c.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_length_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_length_q_mn'] = snow_length_q_mn.astype('float32')     

        if verbose == True:
            print('Added snow season length quality to interannual snowDynamics.')  

    # Best value (requires weighted mean and length)
    if (best_value == True) & (form == 'mean_weighted') & ('length' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_length_bvi = weights_length_c.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        #snow_length_bvi = weights_length_c.fillna(0).argmax(dim = 'winterYear', skipna = True).compute()

        # Get best value from best year
        snow_length_bv = snowDynamics['snow_length'].isel(winterYear = snow_length_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_length_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_length_bv'] = snow_length_bv

        if verbose == True:
            print('Added snow season length best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_length_bvy = xr.where(snow_length_bv.notnull(), snow_length_bvi + 2019, np.nan).astype('float32')
        snow_length_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_length_bvy'] = snow_length_bvy

        if verbose == True:
            print('Added snow season length best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_length_bvq = weights_length_c.isel(winterYear = snow_length_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_length_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_length_bvq'] = snow_length_bvq

            if verbose == True:
                print('Added snow season length best value quality to interannual snowDynamics')       

    # Clear observations per day products (always regular mean)
    if 'year' in clearObsPerDay:

        valid_count = snowDynamics['winterYear_clearObsPerDay'].notnull().sum(dim = 'winterYear')
        snowDynamics['winterYear_clearObsPerDay'] = xr.where(valid_count >= min_count, snowDynamics['winterYear_clearObsPerDay'], np.nan)
        winterYear_clearObsPerDay_mn = snowDynamics['winterYear_clearObsPerDay'].mean(dim = 'winterYear', skipna = True)

        winterYear_clearObsPerDay_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['winterYear_clearObsPerDay_mn'] = winterYear_clearObsPerDay_mn # Add to snowDynamics  

        if verbose == True:
            print('Added mean clear observations per day over the winter year to interannual snowDynamics.')   

    if 'season' in clearObsPerDay:

        valid_count = snowDynamics['snowSeason_clearObsPerDay'].notnull().sum(dim = 'winterYear')
        snowDynamics['snowSeason_clearObsPerDay'] = xr.where(valid_count >= min_count, snowDynamics['snowSeason_clearObsPerDay'], np.nan)
        snowSeason_clearObsPerDay_mn = snowDynamics['snowSeason_clearObsPerDay'].mean(dim = 'winterYear', skipna = True)

        snowSeason_clearObsPerDay_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snowSeason_clearObsPerDay_mn'] = snowSeason_clearObsPerDay_mn # Add to snowDynamics  

        if verbose == True:
            print('Added clear observations per day over the snow season to interannual snowDynamics.')   

    if 'fall' in clearObsPerDay:

        valid_count = snowDynamics['snowFall_clearObsPerDay'].notnull().sum(dim = 'winterYear')
        snowDynamics['snowFall_clearObsPerDay'] = xr.where(valid_count >= min_count, snowDynamics['snowFall_clearObsPerDay'], np.nan)
        snowFall_clearObsPerDay_mn = snowDynamics['snowFall_clearObsPerDay'].mean(dim = 'winterYear', skipna = True)

        snowFall_clearObsPerDay_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snowFall_clearObsPerDay_mn'] = snowFall_clearObsPerDay_mn # Add to snowDynamics  

        if verbose == True:
            print('Added clear observations per day over the snow fall period to interannual snowDynamics.')

    if 'peak' in clearObsPerDay:

        valid_count = snowDynamics['snowPeak_clearObsPerDay'].notnull().sum(dim = 'winterYear')
        snowDynamics['snowPeak_clearObsPerDay'] = xr.where(valid_count >= min_count, snowDynamics['snowPeak_clearObsPerDay'], np.nan)
        snowPeak_clearObsPerDay_mn = snowDynamics['snowPeak_clearObsPerDay'].mean(dim = 'winterYear', skipna = True)

        snowPeak_clearObsPerDay_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snowPeak_clearObsPerDay_mn'] = snowPeak_clearObsPerDay_mn # Add to snowDynamics  

        if verbose == True:
            print('Added clear observations per day over the snow peak period to interannual snowDynamics.')

    if 'melt' in clearObsPerDay:

        valid_count = snowDynamics['snowMelt_clearObsPerDay'].notnull().sum(dim = 'winterYear')
        snowDynamics['snowMelt_clearObsPerDay'] = xr.where(valid_count >= min_count, snowDynamics['snowMelt_clearObsPerDay'], np.nan)
        snowMelt_clearObsPerDay_mn = snowDynamics['snowMelt_clearObsPerDay'].mean(dim = 'winterYear', skipna = True)

        snowMelt_clearObsPerDay_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snowMelt_clearObsPerDay_mn'] = snowMelt_clearObsPerDay_mn # Add to snowDynamics  

        if verbose == True:
            print('Added clear observations per day over the snow melt period to interannual snowDynamics.')

    # Returns xarray dataset with each array as a variable.
    return snowDynamics_i

######################################################################################################################################################

# Convert tiled NetCDF snow dynamics products to wide-area GeoTiff products.
def mergeTiledSnowDynamics(input, output, roi, tiles, input_ID = '', output_ID = '', variables = [], clip = 'minimal', tile_roi = False, 
                           verbose = True):

    '''
    Paramters:
    input (str): Input folder where tiled snow dynamics NetCDF products are saved (path/to/folder/with/files)
    - As saved after dailySnowCube2SnowDynamics() or interannualSnowDynamics()

    output (str): Output folder where region of interest snow dynamics Tif products will be saved (path/to/folder)

    roi (str): Path to wide-area shapefile (path/to/file.shp)
    - Will be used to clip merged product to roi for variables that require it (all but start, end and their u's)
    - Also used to select tiles for merging

    tiles (str): Path to processing tile shapefile (path/to/file.shp)
    - Will be used to filter to NetCDF files overlapping roi 

    input_ID (str): ID from file name in input from which to filter NetCDF files included in merge. '' Should find all NetCDF files. 
    - For example, setting to 'winterYear1819' will only find NetCDF files from that winter year

    output_ID (str): ID to add to file name in output, will be in addition to variable name automatically added. 
    - Format: 'HLS_Fmask_variable_output_ID.tif' (e.g., time_area)

    variables (list of str): List of variable names matching variable names in input NetCDF products.
    - Winter year: 'snow_start', 'snow_end', 'snow_endSum', 'snow_length', 'snow_periods', 'snow_status', 'snow_start_u', 'snow_end_u',  
      'snow_length_u', 'winterYear_clearObsPerDay', 'snowSeason_clearObsPerDay', 'snowFall_clearObsPerDay', 'snowPeak_clearObsPerDay',
      'snowMelt_clearObsPerDay'
    - Interannual: 'snow_start_mn', 'snow_start_sd', 'snow_start_bv', 'snow_end_mn', 'snow_end_sd', 'snow_end_bv', 'snow_endSum_mn', 'snow_endSum_sd',
      'snow_endSum_bv', 'snow_length_mn', 'snow_length_sd', 'snow_length_bv', 'snow_periods_mn', 'snow_periods_sd', 'pPerennialSnow', 'pSnowFree', 
      'snow_start_u_mn', 'snow_start_q_mn', 'snow_start_bvq', 'snow_start_bvy', 'snow_end_u_mn', 'snow_end_q_mn', 'snow_end_bvq', 'snow_end_bvy',
      'snow_endSum_q_mn', 'snow_endSum_bvq', 'snow_endSum_bvy', 'snow_length_u_mn', 'snow_length_q_mn', 'snow_length_bvq', 'snow_length_bvy', 
       'winterYear_clearObsPerDay_mn', 'snowSeason_clearObsPerDay_mn', 'snowFall_clearObsPerDay_mn', 'snowPeak_clearObsPerDay_mn', 
       'snowMelt_clearObsPerDay_mn'
    - Default ([]) will save raster for every variable in dataset

    clip (str): Amount of clipping perform
    - 'none': No clipping.
    - 'minimal': Only clip variables that do not set NaN outside ROI during processing (all but start, end and their u's).
    - 'all': Clip all variables (e.g., if building output from subset of larger ROI).

    tile_roi (bool): Whether (True) or not (False) to apply a small negative buffer to the roi to ensure no extra tiles are included. 
    - Sometimes rois are perfectly aligned with tiles, meaning tiles on outside edge get included. 
    - Note: Clipping is still done with the original roi. 
      
    Returns:
    Saved tifs of all requested snow dynamics products. 
    '''

    # Load tile and roi geodataframes
    roi = gpd.read_file(roi, engine = 'pyogrio')
    tiles = gpd.read_file(tiles, engine = 'pyogrio')

    # Find tiles overlapping roi
    if tile_roi == True: # Slightly reduce size of rois that perfectly match tile boundaries
        tile_list = tiles[tiles.intersects(roi.buffer(-100).unary_union)]
    if tile_roi == False:
        tile_list = tiles[tiles.intersects(roi.unary_union)]
    
    tile_list = tile_list.index.tolist()

    if verbose == True:
        print('Found ' + str(len(tile_list)) + ' tiles overlapping roi.')

    # Find all NetCDF files in folder matching input_ID
    ncs = [nc for nc in glob.glob(os.path.join(input, './**/*.nc'), recursive = True) if input_ID in nc]

    if verbose == True:
        print('Found ' + str(len(ncs)) + ' NetCDF files matching ' + input_ID + '.')

    # Filter to NetCDF files overlapping roi
    ncs_filtered = []

    for nc in ncs:
        for tile in tile_list:
            if str(tile) in nc:
                ncs_filtered.append(nc)

    if verbose == True:
        print('Filtered to ' + str(len(ncs_filtered)) + ' NetCDF files covering tiles overlapping roi.')

    # Before merging with open_mfdataset, need to set function to remove 1-pixel overlap at long 2 edges from dataset
    def remove_edge(ds): 
        #return da[:,1:,0:-1] # Remove north (top) and east (right) edge - flipped for ds slicing
        return ds.isel(x = slice(0, ds.sizes['x'] - 1), y = slice(1, ds.sizes['y'])) 
    
    # Load merged dataset
    merged = xr.open_mfdataset(ncs_filtered, 
                               chunks = 'auto', 
                               parallel = True, 
                               preprocess = remove_edge, 
                               engine = 'netcdf4').astype('float32') # Float32 for NaNs

    # For some reason, interannual open_mfdataset can set spatial_ref as a data variable and remove CRS. This fixes that. 
    if merged.rio.crs == None:
        merged = merged.set_coords('spatial_ref')

    if verbose == True:
        print('Created merged dataset: ' + str(merged.sizes) + ', ' + str(len(merged)) + ' variables.')

    # For all variables of interest, save to raster in specified outputs
    if len(variables) == 0:
        variables = list(merged.keys()) # Name of all data variables as list 

    # For all variables, clip (if asked for and required) and save to GeoTiff
    for variable in variables:

        da = merged[variable] # Single snow dynamic stat from dataset

        if clip == 'minimal': 
            if ('start' not in variable) & (('end' not in variable) | ('Sum' in variable)): # snow_start, snow_end, u's already have outside NaNs
                da = da.rio.clip(roi.geometry.values, drop = False) # Clip to roi and keep same shape with NaNs

        if clip == 'all': # Only clip if roi provided
            da = da.rio.clip(roi.geometry.values, drop = False) # Clip to roi and keep same shape with NaNs

        da.rio.write_nodata(-999, encoded = True, inplace = True) # Ensured NaN NoData values are saved
        da.rio.to_raster(os.path.join(output, 'HLS_Fmask_' + variable + '_' + output_ID + '.tif'))

        if verbose == True:
            print('Product ' + str(variables.index(variable) + 1) + ' of ' + str(len(variables)) + ' saved: ' 
                  + 'HLS_Fmask_' + variable + '_' + output_ID + '.tif')

######################################################################################################################################################