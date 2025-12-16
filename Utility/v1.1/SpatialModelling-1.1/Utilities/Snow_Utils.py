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
import calendar as cal
import subprocess

# Open source
import xarray as xr
import numpy as np
import geopandas as gpd

from rasterio.enums import Resampling
from scipy.signal import find_peaks

# Mine
sys.path.append('C:/Users/mbonney/OneDrive - NRCan RNCan/Projects/UtilityCode/DataAccess/Utilities')
import PreProcess_Utils as pputil

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# mergeTiledSnowDynamics() outputs warnings about large Dask graph sizes (using warnings ignore does not remove, happens on Dask end I think)
# Test impact of xr.set_options(use_bottleneck=False) # For context, got wrong std values in testing... this fixes (but maybe slightly slower?) 
# Can reproject with dask: https://corteva.github.io/rioxarray/html/rioxarray.html (see links here)
# Can save a bit more space (~30%) by saving with smaller dtypes for certain variables (e.g., 16 or 8 instead of 32)
# Update mergeTiledSnowDynamics() to support COGs (see tif_to_cog)
# V2: Test other snow ID systems (e.g., NDSI, SPIRES)
# V2: Can other datasets (e.g., low-res climate, snow) be used to refine metrics, especially when uncertainty is high (even adjust uncertainty)?
# V2: Can machine learning (e.g., random forest) be used to refine metrics, especially when uncertainty is high?
# V2: What adjustments can be made (if needed) for dense canopy cover, mountain slopes, mixed land-water etc? Need national-scale validation. 
# V2: Test impact of other cloud masking methods besides Fmask
# V3?: Consider additional satellites (e.g., Sentinel-1, RCM)

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
    snowCube = xr.concat((cube_yr1, cube_yr2), dim = 'time').sortby('time').chunk({'time': -1, 'x': 501, 'y': 501})
    # Most efficient to chunk here (note for 2001x2001 pixel cube, this is 16 spatial chunks)

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
                            engine = 'h5netcdf',  # Recommended
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

# Inner snow dynamics function that runs on 1D arrays.
def snowDynamics1D(ar, ar_days, sys, sye, syl):

    """
    Parameters:
    ar (numpy array): 1D array of snow values, covering 2 full calendar years of observations.

    ar_days (numpy array): 1D array of days since December 31 (e.g., 0 = Dec 31, 1 = Jan 1, etc.), aligning with ar.

    sys (int): Snow year start day in days since December 31. 

    sye (int): Snow year end day in days since December 31.

    syl (int): Snow year length in days (sye - sys + 1).

    Returns:
    startF (float32): Start day of the FIRST 'snow period', defined as number of days from December 31 in winter year.
    startF_u (float32): Uncertainty (± days) of the FIRST period start date, defined as half the length of the uncertainty period.
    startB (float32): Start day of the BIGGEST snow period.
    startB_u (float32): Uncertainty (± days) of the BIGGEST period start date.
    endL (float32): End day of the LAST 'snow period', defined as number of days from December 31 in winter year.
    endL_u (float32): Uncertainty (± days) of the LAST period end date, defined as half the length of the uncertainty period.
    endB (float32): End day of the BIGGEST snow period.
    endB_u (float32): Uncertainty (± days) of the BIGGEST period end date.
    lengthT (float32): TOTAL number of days with snow cover in the snow-year.
    lengthT_u (float32): Uncertainty (± days) in TOTAL length, defined as half the sum of all uncertainty periods.
    lengthB (float32): Number of days with snow in the BIGGEST snow period.
    lengthB_u (float32): Uncertainty (± days) in the BIGGEST period length.
    periods (float32): Number of snow periods, defined as number of periods of time where snow was observed (separated by non-snow)
    status (float32): Snow status. 0 = Seasonal. 1 = Perennial, 2 = Inconsistent perennial, 3 = Snow Free, 4 = Only ephemeral.
    """
    
    # Required defaults
    startF = np.nan # Start date of first snow period
    startF_u = np.nan # Uncertainty of start date of first snow period
    startB = np.nan # Start date of biggest snow period
    startB_u = np.nan # Uncertainty of start date of biggest snow period

    endL = np.nan # End date of last snow period
    endL_u = np.nan # Uncertainty of end date of last snow period
    endB = np.nan # End date of biggest snow period
    endB_u = np.nan # Uncertainty of end date of biggest snow period    

    # In case all are NaN...
    lengthT = np.nan
    lengthT_u = np.nan
    lengthB = np.nan
    lengthB_u = np.nan
    periods = np.nan
    status = np.nan    

    # Remove NaN observations from ar_days and ar
    ar_days = ar_days[~np.isnan(ar)]
    if len(ar_days) > 0: # If not all time-steps are unclear/outsie ROI...
        ar = ar[~np.isnan(ar)]

        # Generate peak information
        peaks = find_peaks(ar, prominence = 1) # Only need left/right base information

        lengthT = 0 # Total length of time (days) with snow
        lengthT_u = 0 # Uncertainty of length of time with snow
        lengthB = 0 # Length of time with snow during biggest snow period
        lengthB_u = 0 # Uncertainty of time with snow during biggest snow period  

        periods = 0 # Number of snow periods

        # Loop through each find_peaks period, gathering required info from find_peaks output
        for period in range(len(peaks[0])):
            prev0 = peaks[1]['left_bases'][period] # Previous confirmed non-snow observation index
            next0 = peaks[1]['right_bases'][period] # Next confirmed non-snow observation index
            start = int(np.ceil((ar_days[prev0] + ar_days[prev0 + 1]) / 2)) # Start day
            start_u = (ar_days[prev0 + 1] - ar_days[prev0] - 1) / 2 # Start uncertainty
            end = int(np.floor((ar_days[next0] + ar_days[next0 - 1]) / 2)) # End day
            end_u = (ar_days[next0] - ar_days[next0 - 1] - 1) / 2 # End uncertainty
            mid = np.floor((start + end) / 2) # Middle day (days since Dec 31)
            length = end - start + 1 # Number of days

            if (mid >= sys) & (mid <= sye): # Only consider periods in this snow-year
                periods = periods + 1 # Number of snow periods
                lengthT = lengthT + length # Total length with snow
                lengthT_u = lengthT_u + start_u + end_u # Total length uncertainty
                if np.isnan(startF): # First time condition met
                    startF = start # Snow start (first)
                    startF_u = start_u # Start uncertainty (first)
                # Last time condition is met
                endL = end # Snow end (last)
                endL_u = end_u # End uncertainty (last)
                if length > lengthB: # Biggest period length
                    startB = start # Snow start (biggest)
                    startB_u = start_u # Start uncertainty (biggest)
                    endB = end # Snow end (biggest)
                    endB_u = end_u # End uncertainty (biggest)
                    lengthB = length # Biggest length with snow
                    lengthB_u = start_u + end_u # Biggest length uncertainty

        # Double check 0 period scenarios (e.g., snow free, perennial)
        if periods == 0:
        
            # Isolate to just snow-year, plus one edge observation
            ar_bool = (ar_days >= sys) & (ar_days <= sye)
            ar_bool1 = ar_bool.copy()
            ar_bool1[:-1] |= ar_bool[1:]
            ar_bool1[1:] |= ar_bool[:-1]
            ar_days_sy = ar_days[ar_bool1]
            if len(ar_days_sy) > 0: # If not all time-steps are unclear/outsie ROI...
                ar_sy = ar[ar_bool1]

                # Scenario: Perennial snow (All 1s)  
                if ar_sy.mean() == 1:
                    lengthT = syl # u still 0
                    lengthB = syl
                    periods = 1

                # Scenario: Snow to snow, but with non-snow in between
                elif (ar_sy[0] == 1) & (ar_sy[-1] == 1): # Filtered array begins and ends with snow
                    # Get first and last non-snow indices, and calculate start and end
                    ar_sy_bool = (ar_sy == 0)
                    first0 = ar_sy_bool.argmax() 
                    end = int(np.floor((ar_days_sy[first0 - 1] + ar_days_sy[first0]) / 2)) # End day (before start in this scenario)
                    last0 = len(ar_sy) - 1 - ar_sy_bool[::-1].argmax() # Reversed array
                    start =  int(np.ceil((ar_days_sy[last0] + ar_days_sy[last0 + 1]) / 2)) # Start day (after end in this scenario)
                    sy_mid = (sys + sye) / 2 # Middle of snow-year
                    # If end is within snow-year and it is closer to middle than start...
                    if ((end >= sys) & (end <= sye)) & (abs(end - sy_mid) < abs(start - sy_mid)):
                        endL = end
                        endB = end
                        endL_u = (ar_days_sy[first0] - ar_days_sy[first0 - 1] - 1) / 2 # End uncertainty
                        endB_u = endL_u
                        lengthT = end - sys + 1 # Number of days
                        lengthB = lengthT
                        lengthT_u = endL_u
                        lengthB_u = endB_u
                        periods = 1
                    # If start is within snow-year and it is closer to middle than end... (give equal to start)
                    if ((start >= sys) & (start <= sye)) & (abs(start - sy_mid) <= abs(end - sy_mid)):
                        startF = start
                        startB = start
                        startF_u = (ar_days_sy[last0 + 1] - ar_days_sy[last0] - 1) / 2 # Start uncertainty
                        startB_u = startF_u
                        lengthT = sye - start + 1 # Number of days
                        lengthB = lengthT
                        lengthT_u = startF_u
                        lengthB_u = startF_u
                        periods = 1

                # Scenario: Non-snow to snow
                if (ar_sy[0] == 0) & (ar_sy[-1] == 1): # Filtered array begins with non-snow and ends with snow
                    first1 = np.argmax(ar_sy == 1) # First snow
                    start =  int(np.ceil((ar_days_sy[first1 - 1] + ar_days_sy[first1]) / 2)) # Start day
                    # Only not snow free if start is within snow-year or there are snow observations in the snow-year
                    if ((start >= sys) & (start <= sye)) | (ar_sy.sum() > 1):
                        startF = start
                        startB = start
                        startF_u = (ar_days_sy[first1] - ar_days_sy[first1 - 1] - 1) / 2 # Start uncertainty
                        startB_u = startF_u
                        lengthT = sye - start + 1 # Number of days
                        lengthB = lengthT
                        lengthT_u = startF_u
                        lengthB_u = startF_u
                        periods = 1

                # Scenario: Snow to non-snow
                if (ar_sy[0] == 1) & (ar_sy[-1] == 0): # Filtered array begins with snow and ends with non-snow
                    last1 = np.where(ar_sy == 1)[0][-1] # Last snow
                    end = int(np.floor((ar_days_sy[last1 + 1] + ar_days_sy[last1]) / 2)) # End day
                    # Only not snow free if end is within snow-year or there are snow observations in the snow-year
                    if ((end >= sys) & (end <= sye)) | (ar_sy.sum() > 1): 
                        endL = end
                        endB = end
                        endL_u = (ar_days_sy[last1 + 1] - ar_days_sy[last1] - 1) / 2 # End uncertainty
                        endB_u = endL_u
                        lengthT = end - sys + 1 # Number of days
                        lengthB = lengthT
                        lengthT_u = endL_u
                        lengthB_u = endB_u
                        periods = 1
               # Other scenarios are snow free

        # Status        
        test_ar = np.append(ar[(abs(ar_days - sys) <= 7) | (abs(ar_days - sye) <= 7)], 0) # Isolate around snow start and end for status 2
        if lengthT >= syl: # All days in snow-year are snow
            status = 1 # Perennial    
        elif lengthT == 0: # No days in snow-year are snow
            status = 3 # Snow Free  
        elif lengthB <= 7: # Longest snow period is a week or less
            status = 4 # Only ephemeral snow
        elif ((np.nanmax(test_ar) == 1) |
            (startF - sys <= 7) | (sye - endL <= 7) | (np.isnan(startF)) | (np.isnan(endL))): 
            # Within week or past of first/last day of snow-year are snow (or NaN)
            status = 2 # Inconsistent Perennial  
        else: # All other conditions
            status = 0 # Regular Fall/Melt

    return startF, startF_u, startB, startB_u, endL, endL_u, endB, endB_u, lengthT, lengthT_u, lengthB, lengthB_u, periods, status

######################################################################################################################################################

# Converts cleaned snow cube (e.g., from cleanSnowCube()) to output Dataset quantifying various snow dynamics.
def snowCube2SnowDynamics(snowCube_c, yr1_doy = 205, yr2_doy = 205, verbose = True):

    """
    Parameters:
    snowCube_c (xarray dataArray): Cleaned winter year snow cube (e.g., from cleanSnowCube().

    yr1_doy (int): Year 1 day of year for defining snow-year start boundary. Default is 205 (~July 24), which is Canada's IMS snow minimum date.
    yr2_doy (int): Year 2 day of year for defining snow-year end boundary. Default is 205 (~July 24), which is Canada's IMS snow minimum date.

    verbose (bool): Whether (true) or not (false) to print function status.

    Returns:
    snowDynamics: Snow dynamics products (xarray Dataset).
    """

    # Create empty xarray Dataset to fill with selected products
    crs = snowCube_c.rio.crs # Some xarray functions remove crs 
    snowDynamics = xr.Dataset(coords = dict(x = ('x', snowCube_c['x'].values), y = ('y', snowCube_c['y'].values)))
    snowDynamics.rio.write_crs(crs, inplace = True) # Reapply crs

    # Find date information
    yr1 = int(snowCube_c.time.dt.year[0]) # First year represented in cube
    yr2 = int(snowCube_c.time.dt.year[-1]) # Last year represented in cube  
    sys = yr1_doy + 1 - 365 # Snow-year start (days since Dec 31)
    sye = yr2_doy + 1 if cal.isleap(yr2) else yr2_doy # Snow-year end (days since Dec 31)
    syl = sye - sys + 1 # Snow-year length (days)

    # Define days since Dec 31 for all time-steps
    ar_days = (snowCube_c.time.values.astype('datetime64[D]') - np.datetime64(str(yr1) + '-12-31')).astype(int)

    if verbose == True:
        print('Created empty snowDynamics Dataset to fill.')

    # Apply 1D function to xarray
    startF, startF_u, startB, startB_u, endL, endL_u, endB, endB_u, lengthT, lengthT_u, lengthB, lengthB_u, periods, status = xr.apply_ufunc(
        snowDynamics1D,
        snowCube_c, ar_days, sys, sye, syl,
        input_core_dims = [['time'], ['time'], [], [], []],
        output_core_dims = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        vectorize = True,
        dask = 'parallelized',
        output_dtypes = [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, 
                         np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]) # All float32 because of NaNs

    if verbose == True:
        print('Calculated snow dynamics arrays using find_peaks and apply_unfunc')

    # Add to snowDynamics
    snowDynamics['snow_startF'] = startF
    snowDynamics['snow_startF_u'] = startF_u
    snowDynamics['snow_startB'] = startB
    snowDynamics['snow_startB_u'] = startB_u
    snowDynamics['snow_endL'] = endL
    snowDynamics['snow_endL_u'] = endL_u
    snowDynamics['snow_endB'] = endB
    snowDynamics['snow_endB_u'] = endB_u
    snowDynamics['snow_lengthT'] = lengthT
    snowDynamics['snow_lengthT_u'] = lengthT_u
    snowDynamics['snow_lengthB'] = lengthB
    snowDynamics['snow_lengthB_u'] = lengthB_u    
    snowDynamics['snow_periods'] = periods
    snowDynamics['snow_status'] = status

    snowDynamics = snowDynamics.expand_dims(winterYear = xr.Variable('winterYear', [str(yr1) + '-' + str(yr2)]))
    if 'band' in snowDynamics.coords:
        snowDynamics = snowDynamics.reset_coords(names = ['band'], drop = True) # Remove unnessisary 'band' coordinate

    if verbose == True:
        print('Added created arrays to snowDynamics.')

    return snowDynamics

######################################################################################################################################################

# From snowCube2SnowDynamics() output cube, create inter-annual (multi-year) snow dynamics products.
def interannualSnowDynamics(snowDynamics, min_count = 'half', products = ['start', 'end', 'length', 'periods', 'status'], 
                            uncertainty = ['start_u', 'end_u', 'length_u'], form = 'mean_weighted', implausible_snow = [], sd = False, quality = True, 
                            best_value = False, verbose = True):

    """
    Parameters:
    snowDynamics (xarray dataSet): winterYear snow dynamics product produced from snowCube2SnowDynamics()
    - It is expected that snowDynamics is in memory

    min_count (str): How to handle NaNs for each pixel, applies to all products. Supports: 'all', 'half', 'one'.
    - 'all': All winterYears should have a value.
    - 'half': At least half of winterYears should have a value (e.g., At least 3 values over 5 winterYears).
    - 'one': At least one winterYear should have a value.
    - Start and end will have additional NaNs compared to other products in perennial/snow free cases. 

    products (list of str): Snow dynamics products of interest. Supports: ['start', 'end', 'length', 'periods', 'status'].
    - 'start': Start day of first (startF) and biggest (startB) 'snow period'. Impacted by form. 
    - 'end': End day of the last (endL) and biggest (endB) snow period. Impacted by form. 
    - 'length': Number of days with snow cover (total [lengthT] and biggest [lengthB] snow period). Impacted by form. 
    - 'periods': Number of 'snow periods'. Impacted by form. 
    - 'status': Snow status. Interannual: % Years with perennial snow, % Years snow free. 

    uncertainty (list of str): Snow dynamics uncertainty products of interest. Supports: ['start_u', 'end_u', 'length_u'].
    - 'start_u': Uncertainty in start day of first (startF_u) and biggest (startB_u) snow period. Impacted by form. 
    - 'end_u': Uncertainty in end day of last (endL_u) and biggest (endB_u) snow period. Impacted by form. 
    - 'length_u': Total uncertainty (lengthT_u) and biggest snow period uncertainty (lengthB_u). Impacted by form. 

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
            snowDynamics['snow_startF'] = xr.where(snowDynamics['snow_startF_u'] <= 15, snowDynamics['snow_startF'], np.nan) 
            snowDynamics['snow_startB'] = xr.where(snowDynamics['snow_startB_u'] <= 15, snowDynamics['snow_startB'], np.nan)

        valid_count = snowDynamics['snow_startF'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_startF'] = xr.where(valid_count >= min_count, snowDynamics['snow_startF'], np.nan) # Filter by valid_count
        valid_count = snowDynamics['snow_startB'].notnull().sum(dim = 'winterYear')
        snowDynamics['snow_startB'] = xr.where(valid_count >= min_count, snowDynamics['snow_startB'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)     
            snow_startF_mn = snowDynamics['snow_startF'].mean(dim = 'winterYear', skipna = True) # Calculate mean
            snow_startB_mn = snowDynamics['snow_startB'].mean(dim = 'winterYear', skipna = True)

        if form == 'mean_weighted': # Weighted mean (with min_count)

            # Uncertainty (50%)
            weights_uF = 2.718 ** (-0.046 * snowDynamics['snow_startF_u']) # -0.046 means 50% weight on 15, exponential
            weights_uB = 2.718 ** (-0.046 * snowDynamics['snow_startB_u'])

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get doys
                early_start = implausible_snow[1] - 365 # Before this date = early start
                late_startF = int(snowDynamics['snow_startF'].quantile(0.95))  # After this = late start
                late_startB = int(snowDynamics['snow_startB'].quantile(0.95))

                # Get diffs
                early_diffF = abs(snowDynamics['snow_startF'] - early_start) # Negative (before abs) = Before early start
                late_diffF = snowDynamics['snow_startF'] - late_startF # Positive = After late start
                early_diffB = abs(snowDynamics['snow_startB'] - early_start)
                late_diffB = snowDynamics['snow_startB'] - late_startB

                # Isolate outside
                outside_diffF = xr.where(snowDynamics['snow_startF'] >= early_start, 0, early_diffF) # Add early end days
                outside_diffF = xr.where(snowDynamics['snow_startF'] <= late_startF, outside_diffF, late_diffF) # Add late end days
                outside_diffB = xr.where(snowDynamics['snow_startB'] >= early_start, 0, early_diffB) # Add early end days
                outside_diffB = xr.where(snowDynamics['snow_startB'] <= late_startB, outside_diffB, late_diffB) # Add late end days

                # Get weights
                weights_i1F = 2.718 ** (-0.046 * outside_diffF) # -0.046 means 50% weight on 15, exponential
                weights_i1B = 2.718 ** (-0.046 * outside_diffB)

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diffF = abs(snowDynamics['snow_startF'] - snowDynamics['snow_startF'].median(dim = 'winterYear', skipna = True))
            weights_i2F = 2.718 ** (-0.046 * med_diffF) # -0.046 mean 50% weight on 15, exponential
            med_diffB = abs(snowDynamics['snow_startB'] - snowDynamics['snow_startB'].median(dim = 'winterYear', skipna = True))
            weights_i2B = 2.718 ** (-0.046 * med_diffB)

            # Combine weights (weighted sum) and calculate weighted mean
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_start_cF = (weights_uF * 0.5) + (weights_i1F * 0.25) + (weights_i2F * 0.25) # 50% weight for uncertainty, 50% for implausibility
                weights_start_cB = (weights_uB * 0.5) + (weights_i1B * 0.25) + (weights_i2B * 0.25)
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_start_cF = (weights_uF * 0.5) + (0.25) + (weights_i2F * 0.25) # No implausible dates, 0.25 weight by default
                weights_start_cB = (weights_uB * 0.5) + (0.25) + (weights_i2B * 0.25)
            snow_startF_mn = snowDynamics['snow_startF'].weighted(weights_start_cF.fillna(0)).mean(dim = 'winterYear', skipna = True)
            snow_startB_mn = snowDynamics['snow_startB'].weighted(weights_start_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_startF_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_startF_mn'] = snow_startF_mn.astype('float32') # Add to snowDynamics
        snow_startB_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_startB_mn'] = snow_startB_mn.astype('float32')

        if sd == True:
            snow_startF_sd = snowDynamics['snow_startF'].std(dim = 'winterYear', skipna = True)
            snow_startF_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_startF_sd'] = snow_startF_sd # Add to snowDynamics
            snow_startB_sd = snowDynamics['snow_startB'].std(dim = 'winterYear', skipna = True)
            snow_startB_sd.rio.write_crs(crs, inplace = True)
            snowDynamics_i['snow_startB_sd'] = snow_startB_sd

        if verbose == True:
            print('Added snow cover start dates to interannual snowDynamics.')

    # Create snow period end interannual product
    if 'end' in products:

        if form == 'mean_clean': # Only observations wth <= 15 uncertainty
            snowDynamics['snow_endL'] = xr.where(snowDynamics['snow_endL_u'] <= 15, snowDynamics['snow_endL'], np.nan)
            snowDynamics['snow_endB'] = xr.where(snowDynamics['snow_endB_u'] <= 15, snowDynamics['snow_endB'], np.nan)

        valid_count = snowDynamics['snow_endL'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_endL'] = xr.where(valid_count >= min_count, snowDynamics['snow_endL'], np.nan) # Filter by valid_count
        valid_count = snowDynamics['snow_endB'].notnull().sum(dim = 'winterYear')
        snowDynamics['snow_endB'] = xr.where(valid_count >= min_count, snowDynamics['snow_endB'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)   
            snow_endL_mn = snowDynamics['snow_endL'].mean(dim = 'winterYear', skipna = True) # Calculate mean
            snow_endB_mn = snowDynamics['snow_endB'].mean(dim = 'winterYear', skipna = True)

        if form == 'mean_weighted': # Weighted mean (with min_count)

            # Uncertainty (50%)
            weights_uL = 2.718 ** (-0.046 * snowDynamics['snow_endL_u']) # -0.046 means 50% weight on 15, exponential
            weights_uB = 2.718 ** (-0.046 * snowDynamics['snow_endB_u'])

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get doys
                late_end = implausible_snow[0] # After this date = late end
                early_endL = int(snowDynamics['snow_endL'].quantile(0.05)) # Before this = early end
                early_endB = int(snowDynamics['snow_endB'].quantile(0.05))

                # Get diffs
                late_diffL = snowDynamics['snow_endL'] - late_end # Positive = After late end
                early_diffL = abs(snowDynamics['snow_endL'] - early_endL) # Negative (before abs) = Before early end
                late_diffB = snowDynamics['snow_endB'] - late_end
                early_diffB = abs(snowDynamics['snow_endB'] - early_endB)

                # Isolate outside 
                outside_diffL = xr.where(snowDynamics['snow_endL'] >= early_endL, 0, early_diffL) # Add early end days
                outside_diffL = xr.where(snowDynamics['snow_endL'] <= late_end, outside_diffL, late_diffL) # Add late end days
                outside_diffB = xr.where(snowDynamics['snow_endB'] >= early_endB, 0, early_diffB)
                outside_diffB = xr.where(snowDynamics['snow_endB'] <= late_end, outside_diffB, late_diffB)

                # Get weights
                weights_i1L = 2.718 ** (-0.046 * outside_diffL) # -0.046 means 50% weight on 15, exponential
                weights_i1B = 2.718 ** (-0.046 * outside_diffB) 

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diffL = abs(snowDynamics['snow_endL'] - snowDynamics['snow_endL'].median(dim = 'winterYear', skipna = True))  
            weights_i2L = 2.718 ** (-0.046 * med_diffL) # -0.046 mean 50% weight on 15, exponential              
            med_diffB = abs(snowDynamics['snow_endB'] - snowDynamics['snow_endB'].median(dim = 'winterYear', skipna = True))  
            weights_i2B = 2.718 ** (-0.046 * med_diffB)             

            # Combine weights (weighted sum) and calculate weighted mean
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_end_cL = (weights_uL * 0.5) + (weights_i1L * 0.25) + (weights_i2L * 0.25) # 50% weight for uncertainty, 50% for implausibility
                weights_end_cB = (weights_uB * 0.5) + (weights_i1B * 0.25) + (weights_i2B * 0.25)
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_end_cL = (weights_uL * 0.5) + (0.25) + (weights_i2L * 0.25) # No implausible dates, 0.25 weight by default
                weights_end_cB = (weights_uB * 0.5) + (0.25) + (weights_i2B * 0.25)
            snow_endL_mn = snowDynamics['snow_endL'].weighted(weights_end_cL.fillna(0)).mean(dim = 'winterYear', skipna = True)
            snow_endB_mn = snowDynamics['snow_endB'].weighted(weights_end_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_endL_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endL_mn'] = snow_endL_mn.astype('float32') # Add to snowDynamics
        snow_endB_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_endB_mn'] = snow_endB_mn.astype('float32')

        if sd == True:
            snow_endL_sd = snowDynamics['snow_endL'].std(dim = 'winterYear', skipna = True)
            snow_endL_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_endL_sd'] = snow_endL_sd # Add to snowDynamics        
            snow_endB_sd = snowDynamics['snow_endB'].std(dim = 'winterYear', skipna = True)
            snow_endB_sd.rio.write_crs(crs, inplace = True)
            snowDynamics_i['snow_endB_sd'] = snow_endB_sd

        if verbose == True:
            print('Added snow cover end dates to interannual snowDynamics.') 

    # Create snow cover length interannual product
    if 'length' in products:

        if form == 'mean_clean': # Only observations wth <= 30 uncertainty (15 + 15)
            snowDynamics['snow_lengthT'] = xr.where(snowDynamics['snow_lengthT_u'] <= 30, snowDynamics['snow_lengthT'], np.nan)   
            snowDynamics['snow_lengthB'] = xr.where(snowDynamics['snow_lengthB_u'] <= 30, snowDynamics['snow_lengthB'], np.nan) 

        valid_count = snowDynamics['snow_lengthT'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_lengthT'] = xr.where(valid_count >= min_count, snowDynamics['snow_lengthT'], np.nan) # Filter by valid_count
        valid_count = snowDynamics['snow_lengthB'].notnull().sum(dim = 'winterYear')
        snowDynamics['snow_lengthB'] = xr.where(valid_count >= min_count, snowDynamics['snow_lengthB'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count)   
            snow_lengthT_mn = snowDynamics['snow_lengthT'].mean(dim = 'winterYear', skipna = True) # Calculate mean
            snow_lengthB_mn = snowDynamics['snow_lengthB'].mean(dim = 'winterYear', skipna = True)

        if form == 'mean_weighted': # Weighted mean (with min_count)
            
            # Uncertainty (50%)
            weights_uT = 2.718 ** (-0.023 * snowDynamics['snow_lengthT_u']) # -0.023 means 50% weight on 30, exponential
            weights_uB = 2.718 ** (-0.023 * snowDynamics['snow_lengthB_u'])

            # Implausibility 1 (25%): Implausible snow dates
            if len(implausible_snow) == 2:

                # Get lengths
                too_long = 365 - (implausible_snow[1] - implausible_snow[0]) # More than this many days = too much snow cover
                too_shortT = int(snowDynamics['snow_lengthT'].quantile(0.05)) # Less = too little
                too_shortB = int(snowDynamics['snow_lengthB'].quantile(0.05))

                # Get diffs
                long_diffT = snowDynamics['snow_lengthT'] - too_long # Positive = Too long
                short_diffT = abs(snowDynamics['snow_lengthT'] - too_shortT) # Negative (before abs) = Too short
                long_diffB = snowDynamics['snow_lengthB'] - too_long
                short_diffB = abs(snowDynamics['snow_lengthB'] - too_shortB)

                # Isolate outside 
                outside_diffT = xr.where(snowDynamics['snow_lengthT'] <= too_long, 0, long_diffT) # Add too long snow cover
                outside_diffT = xr.where(snowDynamics['snow_lengthT'] >= too_shortT, outside_diffT, short_diffT) # Add too short snow cover
                outside_diffB = xr.where(snowDynamics['snow_lengthB'] <= too_long, 0, long_diffB)
                outside_diffB = xr.where(snowDynamics['snow_lengthB'] >= too_shortB, outside_diffB, short_diffB)

                # Get weights
                weights_i1T = 2.718 ** (-0.046 * outside_diffT) # -0.046 means 50% weight on 15, exponential
                weights_i1B = 2.718 ** (-0.046 * outside_diffB)

            # Implausibility 2 (25%): Implausible snow dynamics variation
            med_diffT = abs(snowDynamics['snow_lengthT'] - snowDynamics['snow_lengthT'].median(dim = 'winterYear', skipna = True))  
            weights_i2T = 2.718 ** (-0.046 * med_diffT) # -0.046 mean 50% weight on 15, exponential    
            med_diffB = abs(snowDynamics['snow_lengthB'] - snowDynamics['snow_lengthB'].median(dim = 'winterYear', skipna = True))
            weights_i2B = 2.718 ** (-0.046 * med_diffB)

            # Combine weights (weighted sum) and calculate weighted mean   
            if len(implausible_snow) == 2: # Using implausible snow weighting
                weights_length_cT = (weights_uT * 0.5) + (weights_i1T * 0.25) + (weights_i2T * 0.25) # 50% weight for uncertainty, 50% for implausibility
                weights_length_cB = (weights_uB * 0.5) + (weights_i1B * 0.25) + (weights_i2B * 0.25)
            if len(implausible_snow) == 0: # Not using implausible snow weighting (e.g., perennial tiles)
                weights_length_cT = (weights_uT * 0.5) + (0.25) + (weights_i2T * 0.25) # No implausible dates, 0.25 weight by default      
                weights_length_cB = (weights_uB * 0.5) + (0.25) + (weights_i2B * 0.25)               
            snow_lengthT_mn = snowDynamics['snow_lengthT'].weighted(weights_length_cT.fillna(0)).mean(dim = 'winterYear', skipna = True)
            snow_lengthB_mn = snowDynamics['snow_lengthB'].weighted(weights_length_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_lengthT_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_lengthT_mn'] = snow_lengthT_mn.astype('float32')# Add to snowDynamics
        snow_lengthB_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_lengthB_mn'] = snow_lengthB_mn.astype('float32')

        if sd == True:
            snow_lengthT_sd = snowDynamics['snow_lengthT'].std(dim = 'winterYear', skipna = True)
            snow_lengthT_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_lengthT_sd'] = snow_lengthT_sd # Add to snowDynamics
            snow_lengthB_sd = snowDynamics['snow_lengthB'].std(dim = 'winterYear', skipna = True)
            snow_lengthB_sd.rio.write_crs(crs, inplace = True)
            snowDynamics_i['snow_lengthB_sd'] = snow_lengthB_sd

        if verbose == True:
            print('Added snow cover length to interannual snowDynamics.') 

    # Create number of snow periods interannual product (always regular mean)
    if 'periods' in products:
  
        valid_count = snowDynamics['snow_periods'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel
        snowDynamics['snow_periods'] = xr.where(valid_count >= min_count, snowDynamics['snow_periods'], np.nan) # Filter by valid_count
        snow_periods_mn = snowDynamics['snow_periods'].mean(dim = 'winterYear', skipna = True) # Calculate mean

        snow_periods_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_periods_mn'] = snow_periods_mn # Add to snowDynamics

        if sd == True:
            snow_periods_sd = snowDynamics['snow_periods'].std(dim = 'winterYear', skipna = True)
            snow_periods_sd.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_periods_sd'] = snow_periods_sd # Add to snowDynamics        

        if verbose == True:
            print('Added snow period count to interannual snowDynamics.') 

    # Create snow status interannual product (0 = Seasonal, 1 = Perennial, 2  = Inconsistent Perennial, 3 = Snow Free, 4 = Only ephemeral)
    if 'status' in products: # Note: status not impacted by mean rules     
        valid_count = snowDynamics['snow_status'].notnull().sum(dim = 'winterYear') # Number of valid values per pixel

        # % years with perennial snow  
        pPerennialSnow = xr.where(snowDynamics['snow_status'] == 2, 1, snowDynamics['snow_status']) # 2 becomes 1 (1 already 1)
        pPerennialSnow = xr.where(pPerennialSnow >= 3, 0, pPerennialSnow) # 3 and 4 becomes 0
        pPerennialSnow = (pPerennialSnow.sum(dim = 'winterYear', min_count = min_count) # Sum (pixels reaching min_count)
                          / valid_count * 100) # Divided by # winterYears with value * 100
        pPerennialSnow.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['pPerennialSnow'] = pPerennialSnow.astype('float32') # Add to snowDynamics

        # % years snow-free
        pSnowFree = xr.where((snowDynamics['snow_status'] == 2) | ((snowDynamics['snow_status'] == 1)), 0, snowDynamics['snow_status']) # 2/1 become 0
        pSnowFree = xr.where((pSnowFree == 3) | (pSnowFree == 4), 1, pSnowFree) # 3 and 4 becomes 1
        pSnowFree = (pSnowFree.sum(dim = 'winterYear', min_count = min_count) # Sum (pixels reaching min_count)
                     / valid_count * 100) # Divided by # winterYears with value * 100  
        pSnowFree.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['pSnowFree'] = pSnowFree.astype('float32')  # Add to snowDynamics               

        if verbose == True:
            print('Added % years snow-free and with perennial snow to snowDynamics.')

    # Start: uncertainty, quality, best value and year
    # Uncertainty (requires start)
    if ('start' in products) & ('start_u' in uncertainty):

        snowDynamics['snow_startF_u'] = xr.where(snowDynamics['snow_startF'].notnull(), snowDynamics['snow_startF_u'], np.nan)
        snowDynamics['snow_startB_u'] = xr.where(snowDynamics['snow_startB'].notnull(), snowDynamics['snow_startB_u'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count), using NaNs from snow_start to clean
            snow_startF_u_mn = snowDynamics['snow_startF_u'].mean(dim = 'winterYear', skipna = True) # Calculate mean
            snow_startB_u_mn = snowDynamics['snow_startB_u'].mean(dim = 'winterYear', skipna = True)

        if form == 'mean_weighted': # Weighted mean (with min_count), can use weights from above
            snow_startF_u_mn = snowDynamics['snow_startF_u'].weighted(weights_start_cF.fillna(0)).mean(dim = 'winterYear', skipna = True)
            snow_startB_u_mn = snowDynamics['snow_startB_u'].weighted(weights_start_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_startF_u_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_startF_u_mn'] = snow_startF_u_mn.astype('float32') # Add to snowDynamics  
        snow_startB_u_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_startB_u_mn'] = snow_startB_u_mn.astype('float32')

        if verbose == True:
            print('Added snow cover start date uncertainty to interannual snowDynamics.') 

    # Quality (requires weighted mean and start)
    if (quality == True) & (form == 'mean_weighted') & ('start' in products): 
        snow_startF_q_mn = weights_start_cF.weighted(weights_start_cF.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_startF_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_startF_q_mn'] = snow_startF_q_mn.astype('float32')
        snow_startB_q_mn = weights_start_cB.weighted(weights_start_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_startB_q_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_startB_q_mn'] = snow_startB_q_mn.astype('float32')

        if verbose == True:
            print('Added snow cover start dates quality to interannual snowDynamics.')

    # Best value (requires weighted mean and start)
    if (best_value == True) & (form == 'mean_weighted') & ('start' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_startF_bvi = weights_start_cF.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        snow_startB_bvi = weights_start_cB.fillna(0).argmax(dim = 'winterYear', skipna = True)

        # Get best value from best year
        snow_startF_bv = snowDynamics['snow_startF'].isel(winterYear = snow_startF_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_startF_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_startF_bv'] = snow_startF_bv
        snow_startB_bv = snowDynamics['snow_startB'].isel(winterYear = snow_startB_bvi).reset_coords(names = 'winterYear', drop = True)
        snow_startB_bv.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_startB_bv'] = snow_startB_bv

        if verbose == True:
            print('Added snow cover start dates best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_startF_bvy = xr.where(snow_startF_bv.notnull(), snow_startF_bvi + 2019, np.nan).astype('float32')
        snow_startF_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_startF_bvy'] = snow_startF_bvy
        snow_startB_bvy = xr.where(snow_startB_bv.notnull(), snow_startB_bvi + 2019, np.nan).astype('float32')
        snow_startB_bvy.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_startB_bvy'] = snow_startB_bvy

        if verbose == True:
            print('Added snow cover start dates best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_startF_bvq = weights_start_cF.isel(winterYear = snow_startF_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_startF_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_startF_bvq'] = snow_startF_bvq.astype('float32')
            snow_startB_bvq = weights_start_cB.isel(winterYear = snow_startB_bvi).reset_coords(names = 'winterYear', drop = True) 
            snow_startB_bvq.rio.write_crs(crs, inplace = True)
            snowDynamics_i['snow_startB_bvq'] = snow_startB_bvq.astype('float32')

            if verbose == True:
                print('Added snow cover start dates best value quality to interannual snowDynamics')

    # End: uncertainty, quality, best value and year
    # Uncertainty (requires end)
    if (('end' in products) & ('end_u' in uncertainty)):

        snowDynamics['snow_endL_u'] = xr.where(snowDynamics['snow_endL'].notnull(), snowDynamics['snow_endL_u'], np.nan)
        snowDynamics['snow_endB_u'] = xr.where(snowDynamics['snow_endB'].notnull(), snowDynamics['snow_endB_u'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count), using NaNs from snow_end to clean
            snow_endL_u_mn = snowDynamics['snow_endL_u'].mean(dim = 'winterYear', skipna = True) # Calculate mean
            snow_endB_u_mn = snowDynamics['snow_endB_u'].mean(dim = 'winterYear', skipna = True)

        if form == 'mean_weighted': # Weighted mean (with min_count), can use weights from above
            snow_endL_u_mn = snowDynamics['snow_endL_u'].weighted(weights_end_cL.fillna(0)).mean(dim = 'winterYear', skipna = True)
            snow_endB_u_mn = snowDynamics['snow_endB_u'].weighted(weights_end_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_endL_u_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endL_u_mn'] = snow_endL_u_mn.astype('float32') # Add to snowDynamics
        snow_endB_u_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_endB_u_mn'] = snow_endB_u_mn.astype('float32')

        if verbose == True:
            print('Added snow cover end dates uncertainty to interannual snowDynamics.') 

    # Quality (requires weighted mean and end)
    if (quality == True) & (form == 'mean_weighted') & ('end' in products): 
        snow_endL_q_mn = weights_end_cL.weighted(weights_end_cL.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_endL_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endL_q_mn'] = snow_endL_q_mn.astype('float32')
        snow_endB_q_mn = weights_end_cB.weighted(weights_end_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_endB_q_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_endB_q_mn'] = snow_endB_q_mn.astype('float32')

        if verbose == True:
            print('Added snow cover end dates quality to interannual snowDynamics.')   

    # Best value (requires weighted mean and end)
    if (best_value == True) & (form == 'mean_weighted') & ('end' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_endL_bvi = weights_end_cL.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        snow_endB_bvi = weights_end_cB.fillna(0).argmax(dim = 'winterYear', skipna = True)

        # Get best value from best year
        snow_endL_bv = snowDynamics['snow_endL'].isel(winterYear = snow_endL_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_endL_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endL_bv'] = snow_endL_bv
        snow_endB_bv = snowDynamics['snow_endB'].isel(winterYear = snow_endB_bvi).reset_coords(names = 'winterYear', drop = True)
        snow_endB_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endB_bv'] = snow_endB_bv

        if verbose == True:
            print('Added snow cover end dates best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_endL_bvy = xr.where(snow_endL_bv.notnull(), snow_endL_bvi + 2019, np.nan).astype('float32')
        snow_endL_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_endL_bvy'] = snow_endL_bvy
        snow_endB_bvy = xr.where(snow_endB_bv.notnull(), snow_endB_bvi + 2019, np.nan).astype('float32')
        snow_endB_bvy.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_endB_bvy'] = snow_endB_bvy

        if verbose == True:
            print('Added snow cover end dates best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_endL_bvq = weights_end_cL.isel(winterYear = snow_endL_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_endL_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_endL_bvq'] = snow_endL_bvq.astype('float32')
            snow_endB_bvq = weights_end_cB.isel(winterYear = snow_endB_bvi).reset_coords(names = 'winterYear', drop = True)
            snow_endB_bvq.rio.write_crs(crs, inplace = True)
            snowDynamics_i['snow_endB_bvq'] = snow_endB_bvq.astype('float32')

            if verbose == True:
                print('Added snow cover end dates best value quality to interannual snowDynamics')

    # Length: uncertainty, quality, best value and year
    # Uncertainty (requires length)
    if ('length' in products) &  ('length_u' in uncertainty): 

        snowDynamics['snow_lengthT_u'] = xr.where(snowDynamics['snow_lengthT'].notnull(), snowDynamics['snow_lengthT_u'], np.nan)
        snowDynamics['snow_lengthB_u'] = xr.where(snowDynamics['snow_lengthB'].notnull(), snowDynamics['snow_lengthB_u'], np.nan)

        if (form == 'mean') | (form == 'mean_clean'): # Regular mean (with min_count), using NaNs from snow_length to clean
            snow_lengthT_u_mn = snowDynamics['snow_lengthT_u'].mean(dim = 'winterYear', skipna = True) # Calculate mean
            snow_lengthB_u_mn = snowDynamics['snow_lengthB_u'].mean(dim = 'winterYear', skipna = True)

        if form == 'mean_weighted': # Weighted mean (with min_count), can use weights from above
            snow_lengthT_u_mn = snowDynamics['snow_lengthT_u'].weighted(weights_length_cT.fillna(0)).mean(dim = 'winterYear', skipna = True)
            snow_lengthB_u_mn = snowDynamics['snow_lengthB_u'].weighted(weights_length_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)

        snow_lengthT_u_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_lengthT_u_mn'] = snow_lengthT_u_mn.astype('float32') # Add to snowDynamics
        snow_lengthB_u_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_lengthB_u_mn'] = snow_lengthB_u_mn.astype('float32')

        if verbose == True:
            print('Added snow season length uncertainty to interannual snowDynamics.')   

    # Quality (requires weighted mean and length)
    if (quality == True) & (form == 'mean_weighted') & ('length' in products): 
        snow_lengthT_q_mn = weights_length_cT.weighted(weights_length_cT.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_lengthT_q_mn.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_lengthT_q_mn'] = snow_lengthT_q_mn.astype('float32')
        snow_lengthB_q_mn = weights_length_cB.weighted(weights_length_cB.fillna(0)).mean(dim = 'winterYear', skipna = True)
        snow_lengthB_q_mn.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_lengthB_q_mn'] = snow_lengthB_q_mn.astype('float32')

        if verbose == True:
            print('Added snow season length quality to interannual snowDynamics.')  

    # Best value (requires weighted mean and length)
    if (best_value == True) & (form == 'mean_weighted') & ('length' in products): 

        # Calculate best value index (winter year index with highest weight)
        snow_lengthT_bvi = weights_length_cT.fillna(0).argmax(dim = 'winterYear', skipna = True) # Fill na with 0 weight or get error
        snow_lengthB_bvi = weights_length_cB.fillna(0).argmax(dim = 'winterYear', skipna = True)

        # Get best value from best year
        snow_lengthT_bv = snowDynamics['snow_lengthT'].isel(winterYear = snow_lengthT_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
        snow_lengthT_bv.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_lengthT_bv'] = snow_lengthT_bv
        snow_lengthB_bv = snowDynamics['snow_lengthB'].isel(winterYear = snow_lengthB_bvi).reset_coords(names = 'winterYear', drop = True)
        snow_lengthB_bv.rio.write_crs(crs, inplace = True)
        snowDynamics_i['snow_lengthB_bv'] = snow_lengthB_bv

        if verbose == True:
            print('Added snow season length best value to interannual snowDynamics')   

        # Adjust best value year (add 2019 to get second year of winter year for output, and set NaNs)
        snow_lengthT_bvy = xr.where(snow_lengthT_bv.notnull(), snow_lengthT_bvi + 2019, np.nan).astype('float32')
        snow_lengthT_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_lengthT_bvy'] = snow_lengthT_bvy
        snow_lengthB_bvy = xr.where(snow_lengthB_bv.notnull(), snow_lengthB_bvi + 2019, np.nan).astype('float32')
        snow_lengthB_bvy.rio.write_crs(crs, inplace = True) # Reapply crs
        snowDynamics_i['snow_lengthB_bvy'] = snow_lengthB_bvy

        if verbose == True:
            print('Added snow season length best value year to interannual snowDynamics')

        # Get best value quality (weight)
        if quality == True:
            snow_lengthT_bvq = weights_length_cT.isel(winterYear = snow_lengthT_bvi).reset_coords(names = 'winterYear', drop = True) # Don't need new coord
            snow_lengthT_bvq.rio.write_crs(crs, inplace = True) # Reapply crs
            snowDynamics_i['snow_lengthT_bvq'] = snow_lengthT_bvq.astype('float32')
            snow_lengthB_bvq = weights_length_cB.isel(winterYear = snow_lengthB_bvi).reset_coords(names = 'winterYear', drop = True)
            snow_lengthB_bvq.rio.write_crs(crs, inplace = True)
            snowDynamics_i['snow_lengthB_bvq'] = snow_lengthB_bvq.astype('float32')

            if verbose == True:
                print('Added snow season length best value quality to interannual snowDynamics')       

    # Returns xarray dataset with each array as a variable.
    return snowDynamics_i

######################################################################################################################################################

# Convert tiled NetCDF snow dynamics products to wide-area COG products.
def mergeTiledSnowDynamics(input, output, roi, tiles, input_ID = '', output_ID = '', clip = 'all', tile_roi = False, verbose = True):

    '''
    Paramters:
    input (str): Input folder where tiled snow dynamics NetCDF products are saved (path/to/folder/with/files)
    - As saved after snowCube2SnowDynamics() or interannualSnowDynamics()

    output (str): Output folder where region of interest snow dynamics Tif products will be saved (path/to/folder)

    roi (str): Path to wide-area shapefile (path/to/file.shp)
    - Will be used to clip merged product to roi (if needed)
    - Also used to select tiles for merging

    tiles (str): Path to processing tile shapefile (path/to/file.shp)
    - Will be used to filter to NetCDF files overlapping roi 

    input_ID (str): ID from file name in input from which to filter NetCDF files included in merge. '' Should find all NetCDF files. 
    - For example, setting to 'winterYear1819' will only find NetCDF files from that winter year

    output_ID (str): ID to add to file name in output, will be in addition to variable name automatically added. 
    - Format: 'HLS_Fmask_variable_output_ID.tif' (e.g., time_area)

    clip (str): Amount of clipping perform
    - 'none': No clipping.
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
        tile_list = tiles[tiles.intersects(roi.buffer(-100).union_all())]
    if tile_roi == False:
        tile_list = tiles[tiles.intersects(roi.union_all())]
    
    tile_list = tile_list.index.tolist()

    tile_list = [f'_{tile}.' for tile in tile_list] + [f'_{tile}_' for tile in tile_list] # Get exact tile names required

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
            if tile in nc:
                ncs_filtered.append(nc)

    if verbose == True:
        print('Filtered to ' + str(len(ncs_filtered)) + ' NetCDF files covering tiles overlapping roi.')

    # Before merging with open_mfdataset, need to set function to remove 1-pixel overlap at long 2 edges from dataset
    def remove_edge(ds): 
        #return da[:,1:,0:-1] # Remove north (top) and east (right) edge - flipped for ds slicing
        return ds.isel(x = slice(0, ds.sizes['x'] - 1), y = slice(1, ds.sizes['y'])) 
    
    # Load merged dataset
    # Some good notes here on potential issues with open_mfdataset: https://github.com/pydata/xarray/discussions/8925
    merged = xr.open_mfdataset(ncs_filtered, 
                               chunks = 'auto', 
                               parallel = False, # True leads to missing/wrong location tiles in some cases
                               preprocess = remove_edge, 
                               engine = 'h5netcdf')

    # For some reason, interannual open_mfdataset can set spatial_ref as a data variable and remove CRS. This fixes that. 
    if merged.rio.crs == None:
        merged = merged.set_coords('spatial_ref')

    if verbose == True:
        print('Created merged dataset: ' + str(merged.sizes) + ', ' + str(len(merged)) + ' variables.')

    # For all variables, clip (if asked for and required) and save to GeoTiff
    for variable in merged.data_vars:

        input_tmp = os.path.join(output, f"{variable}_temp.tif")
        filename = 'HLS_Fmask_v1_1_' + variable + '_' + output_ID + '.tif'
        output_cog = os.path.join(output, filename)

        # Skip if file already exists
        if os.path.exists(output_cog):
            print(f"Skipping existing file: {output_cog}")
            continue

        if verbose == True:    
            print(f"Processing variable: {variable}")

        if clip == 'all': # Only clip if roi provided
            da = da.rio.clip(roi.geometry.values, drop = False) # Clip to roi and keep same shape with NaNs

        # Step 1. Write variable to temporary GeoTIFF (streaming write)
        da = merged[variable] # Single snow dynamic stat from dataset
        da.rio.write_nodata(-32767, encoded = True, inplace = True) # Ensured NaN NoData values are saved (-32767 is datacube standard)
        if da['y'][0].item() < da['y'][-1].item(): # If coordinates are flipped
            da = da.sortby('y', ascending = False)
        da.rio.to_raster(input_tmp,
                driver = "GTiff",
                tiled = True,
                blockxsize = 512, 
                blockysize = 512,
                bigtiff = "YES",
                compress = "LZW",
                dtype = str(da.dtype),
                num_threads = "ALL_CPUS")
        
        # Step 2. Define resampling and timestamp metadata based on variable name
        resampling = 'average' # Interannual always uses average and most winter year uses average
        if '1824' in filename: 
            datetime = '2024:07:01 00:00:00' # Interannual, so use July 1 of last year
        else:
            datetime = '20' + filename.rsplit('_', 1)[0][-4:-2] + ':12:31 00:00:00' # Dec 31 of first year for winter year
            if ('periods' in filename) or ('status' in filename): 
                resampling = 'mode' # More categorical metrics    

        # Step 3. Build GDAL translate command
        cmd = ["gdal_translate",
                "-of", "COG",
                "-co", "COMPRESS=LZW",
                "-co", f"OVERVIEW_RESAMPLING={resampling}",
                "-co", "NUM_THREADS=ALL_CPUS",
                "-co", "BIGTIFF=YES",
                input_tmp,
                output_cog]
        cmd += ["-mo", f"TIFFTAG_DATETIME={datetime}"]

        # Step 4. Run GDAL translate safely
        try:
            subprocess.run(cmd, check = True, shell = True)
            if verbose == True:
                print(f"✅ Created COG: {output_cog}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting {variable} to COG: {e}")

        # Step 5. Cleanup temporary file
        os.remove(input_tmp)

######################################################################################################################################################