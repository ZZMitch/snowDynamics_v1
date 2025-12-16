######################################################################################################################################################
#
#   name:       Viz_Utils.py
#   contains:   Functions for visualizing data (e.g., with matplotlib, hvplot and similar Python tools)
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Open source
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import polars as pl
import xarray as xr
import statsmodels.api as sm
import pandas as pd

from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.cm import ScalarMappable

# Built in
import datetime as dt

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Add intereactive plotting with hvplot (see VizTool in old code...) - Polars integrates with hvplot using default .plot()
# Add plots from HLS Canada observation availability project
# stackedBar() requires dataarrays with matching unique values

######################################################################################################################################################
# Maps ###############################################################################################################################################
######################################################################################################################################################

# Creates a plot matrix from xarray data cube.
def matrixPlot(cube, form, sat = '', byCon = True, bands = ['RED', 'GREEN', 'BLUE'], range = [], save = ''):

    """
    Parameters:
    cube (in-memory xarray): Cube of interest for plotting

    form (str): Form of data being plotted. Supports: 'spectral', 'Fmask' (10-cat), 'ims' (4-cat), 
    'ims_dynamics' (start, end, length, periods, status), 'ims_pSnow (monthly)', . 

    sat (str): Satellite constellation being plotted. Supports: 'hls', 's2'. '' = Not a satellite. 

    byCon (bool): If multiple satellite constellations included in HLS, adjust time coordinates to note this

    bands (list of str): Spectral bands to plot in RGB format. Default is true color. Only applies to 'spectral' form.

    range (list of two num): Range of color bar in var units.  Defaults from min to max. Supports: 'spectral', 'ims_dynamics'.
    - spectral: Recommend [0, X] (0.15 for RGB. 0.5 if NIR included. 0.35 if SWIR included but not NIR)
    - ims_dynamics: Recommend [0, 365] (length)
    - ims_pSnow: Recommend [0, 100]

    save (str): Save figure as file name (e.g., 'fig.tif'). '': Don't save. 

    Returns:
    plot matrix of form requested
    """

    if sat == 'hls': 
        cube1 = adjustTime(cube, '', byCon = byCon)
    if sat == 's2':
        cube1 = adjustTime(cube, '', byCon = False)

    if form == 'spectral':

        if range == []:
            range = [cube.min(), cube.max()]

        plot = cube1.sel(band = bands).plot.imshow(col = 'time', col_wrap = 5, size = 3.5, vmin = range[0], vmax = range[1]) # aspect = 1

        plt.tight_layout(h_pad = 0.25, w_pad = -12.5)

    if form == 'Fmask':

        if save == '':
            div = 1
            h_pad = 0.25
            w_pad = -12.5
            size = 12
            anchor = -5.5
        if save != '':
            div = 2.175
            h_pad = -1.5
            w_pad = -15
            size = 8
            anchor = -3.5

        colors = ['#FF0000', '#4DA91C', '#1C6BA9', '#17FFFB', '#649749', '#497497', '#46c6c4', '#d3d3d3', '#808080', '#FFFFFF', '#000000']
        scale = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(scale, len(colors))

        plot = cube1.squeeze().plot.imshow(col = 'time', col_wrap = 5, size = 3.5 / div, cmap = cmap, norm = norm, add_colorbar = False)

        plt.tight_layout(h_pad = h_pad, w_pad = w_pad)

        # Set up colorbar (note: looks best in scenarios where number of time-steps are divisible by 5, -1 (e.g., 14))
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = plot.axs[-1,-1], anchor = (anchor, 0.5), 
                          ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cb.ax.tick_params(size = 0, labelsize = size)
        cb.ax.set_yticklabels(['Outside ROI', 'Land', 'Water', 'Snow', 'Land (High Aerosol)', 'Water (High Aerosol)', 'Snow (High Aerosol)', 
                               'Cloud-adjacent', 'Shadow', 'Cloud', 'Fill'])

    if form == 'ims':

        if save == '':
            div = 1
            h_pad = 0.25
            w_pad = 0
            size = 12
            anchor = -5.5
        if save != '':
            div = 2.325
            h_pad = -1.75
            w_pad = -11.5
            size = 8
            anchor = -3.5

        colors = ['#FFFFFF', '#002573', '#267300', '#0099FF', '#BFBFBF']
        scale = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(scale, len(colors))  

        plot = cube.squeeze().plot.imshow(col = 'time', col_wrap = 5, size = 3.5 / div, cmap = cmap, norm = norm, add_colorbar = False) 

        plt.tight_layout(h_pad = h_pad, w_pad = w_pad)

        # Set up colorbar (note: looks best in scenarios where number of time-steps are divisible by 5, -1 (e.g., 14))
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = plot.axs[-1,-1], anchor = (anchor, 0.5), 
                          ticks = [0, 1, 2, 3, 4])
        cb.ax.tick_params(size = 0, labelsize = size)
        cb.ax.set_yticklabels(['Outside ROI', 'Water', 'Land', 'Ice', 'Snow'])

    if form == 'ims_dynamics':

        if range == []:
            range = [cube.min(), cube.max()]

        plot = cube.plot.imshow(col = 'winterYear', col_wrap = 5, size = 3.5, cmap = 'viridis', vmin = range[0], vmax = range[1], 
                                cbar_kwargs = {'pad': 0.01, 'shrink': 0.9}) # aspect = 1
        
        plt.tight_layout(h_pad = 0.25, w_pad = -12.5)
        
    if form == 'ims_pSnow':    

        if range == []:
            range = [cube.min(), cube.max()]

        plot = cube.plot.imshow(col = 'month', col_wrap = 3, size = 3.5, cmap = 'viridis', vmin = range[0], vmax = range[1], 
                                cbar_kwargs = {'anchor': (0.01, 5.325), 'shrink': 0.18, 'orientation': 'horizontal', 'label': ''}) 
        plot.axs[0, 0].annotate(cube.name, (-2200000, -140000)) # Map coordinates, could not get other forms to show title

        month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        for img, title in zip(plot.axs.flat, month_list):
            img.set_title(title, pad = 3)

        plt.subplots_adjust(hspace = -0.05, wspace = 0.01) # Adjust space between subplots

    for i, img in enumerate(plot.axs.flat):
        img.set_yticks([]) # Remove y axis ticks
        img.set_xticks([]) # Remove x axis ticks
        img.set_ylabel('') # Remove y axis labels
        img.set_xlabel('') # Remove x axis labels
        img.set_aspect('equal') # Force square pixels
        if (form == 'Fmask') or (form == 'ims'):
            img.set_title(img.title.get_text()[7:], fontsize = size, pad = 1) # Remotes 'time = ' 

    plt.show()

    # Save
    if save != '':
        plot.fig.get_figure().savefig(save, dpi = 600, bbox_inches = 'tight', pil_kwargs = {'compression': 'tiff_lzw'})

######################################################################################################################################################        

# Adjusts time coordinates of xarray cube for plotting, including adding satellite.
def adjustTime(cube, median, byCon):   

    """
    Parameters: 
    cube (in-memory xarray): Cube of interest for plotting

    median (str): If cube has been resampled to some time interval, adjusts time coordinates to match

    byCon (bool): If multiple satellite constellations included, adjust time coordinates to note this

    Returns:
    cube with adjusted time coordinates
    """

    cube1 = cube.copy() # Avoid time in original cube getting messed up

    if type(cube1['time'].values.dtype) == np.str_: # If this was already run
        return cube # Skip

    if median == '':
        cube1['time'] = cube1['time'].dt.strftime('%Y-%m-%d')
        if byCon == True:
            cube1['time'] = ('time', np.char.add(cube1['time'].astype(str).data, ' / '))
            cube1['time'] = ('time', np.char.add(cube1['time'].data, cube1['constellation'].data))
            
    if median == 'monthly':
        cube1['time'] = cube1['time'].dt.strftime('%Y-%m')
        cube1['time'] = ('time', np.char.add(cube1['time'].astype(str).data, ' Monthly Median'))
        
    if median == 'seasonal':
        cube1['time'] = cube1['time'].dt.strftime('%Y-%m')
        cube1['time'] = ('time', np.char.add(cube1['Date'].astype(str).data, ' Seasonal Median'))
    
    if median == 'yearly':
        cube1['time'] = cube1['time'].dt.strftime('%Y')
        cube1['time'] = ('time', np.char.add(cube1['time'].astype(str).data, ' Yearly Median'))
    
    return cube1

######################################################################################################################################################

# Map a single variable from a raster (e.g., dataArray) or vector (gdf, e.g., tiles) over Canada
def mapCanada(data, ax = None, form = '', var = '', label = '', range = [], cmap = 'viridis', vcon1 = [], con1_col = '', con1_lab = '', vcon2 = [], 
              con2_col = '', con2_lab = '', vcon3 = [], con3_col = '', con3_lab = '', vcon4 = [], con4_col = '', con4_lab = '', single_plot = True):

    """
    Parameters:
    data (vector or raster): Vector or raster data to map. 

    ax (mpl axis): Axis to plot on.
    
    form (str): Form of data to map. Supports: 'vector', 'raster'.

    var (str): Name of variable to map across all tiles. 

    label (str): Long-form descriptor to put on map for var.

    range (list of two num): Range of color bar in var units.  Defaults from min to max. 

    cmap (str): Color map to use var. 

    vcon1 (gdf): First conditional subset of vector representing categorical data to map. 

    con1_col (str): Color of vcon1 on map. 

    con1_lab (str): vcon1 label in legend. 

    vcon2 (gdf): Second conditional subset of vector representing categorical data to map. 

    con2_col (str): Color of vcon2 on map. 

    con2_lab (str): vcon2 label in legend. 

    vcon3 (gdf): Third conditional subset of vector representing categorical data to map. 

    con3_col (str): Color of vcon3 on map. 

    con3_lab (str): vcon3 label in legend. 

    vcon4 (gdf): Fourth conditional subset of vector representing categorical data to map. 

    con4_col (str): Color of vcon4 on map. 

    con4_lab (str): vcon4 label in legend. 

    single_plot (bool): True = Single plot, False = Subplot. 

    Returns:
    Map of varible over Canada. 
    """

    if single_plot == True: 
        fig, ax = plt.subplots(figsize = (10, 7.5))
    ax = ax or plt.gca()

    if form == 'vector':

        if range == []: # Default range = min to max    
            range = [data[var].min(), data[var].max()]

        # Main plot
        main = data.plot(data[var], vmin = range[0], vmax = range[1], cmap = cmap, edgecolor = 'black', linewidth = 0.25, ax = ax)

        # Conditional plots
        handles = []

        if type(vcon1) != list: # Condition 1
            con1 = vcon1.plot(color = con1_col, edgecolor = 'black', linewidth = 0.25, ax = ax, label = con1_lab)
            con1_patch = mpatches.Patch(facecolor = con1_col, edgecolor = 'black', label = con1_lab)
            handles.append(con1_patch)
        if type(vcon2) != list: # Condition 2
            con2 = vcon2.plot(color = con2_col, edgecolor = 'black', linewidth = 0.25, ax = ax, label = con2_lab)
            con2_patch = mpatches.Patch(facecolor = con2_col, edgecolor = 'black', label = con2_lab)
            handles.append(con2_patch)
        if type(vcon3) != list: # Condition 3
            con3 = vcon3.plot(color = con3_col, edgecolor = 'black', linewidth = 0.25, ax = ax, label = con3_lab)
            con3_patch = mpatches.Patch(facecolor = con3_col, edgecolor = 'black', label = con3_lab)
            handles.append(con3_patch)
        if type(vcon4) != list: # Condition 4
            con4 = vcon4.plot(color = con4_col, edgecolor = 'black', linewidth = 0.25, ax = ax, label = con4_lab)
            con4_patch = mpatches.Patch(facecolor = con4_col, edgecolor = 'black', label = con4_lab)
            handles.append(con4_patch)

        if single_plot == True: 
            plt.legend(handles = handles)

    if form == 'raster':

        if range == []: # Default range = min to max    
            range = [data.min(), data.max()]

        # Main plot
        main = data.plot(vmin = range[0], vmax = range[1], cmap = cmap, ax = ax, add_colorbar = False, add_labels = False)

    # Hide  axis ticks and labels
    ax.xaxis.set_tick_params(labelbottom = False)
    ax.yaxis.set_tick_params(labelleft = False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal') # Force square pixels

    fig = ax.get_figure() 

    # Configure colorbar
    if single_plot == True: 
        cbax = fig.add_axes([0.2, 0.15, 0.36, 0.03]) # Left, Bottom, Width, Height
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = range[0], vmax = range[1]))
        fig.colorbar(sm, cax = cbax, orientation = 'horizontal')
        cbax.set_xlabel(label, labelpad = -47)

    #plt.show()

    if single_plot == False:
        return fig

######################################################################################################################################################

# Map Canada-wide snow season start and end dates as easy-to-interpret categories.
def mapCanadaSnowStartEnd(array, form, pPerennial, pSnowFree, ax = None, single_plot = True):

    """
    Parameters:
    array (dataArray): DataArray containing start and end dates to map. 

    form (str): Form of array. Supports 'start' (snow start date), 'end' (snow end date)

    pPerennial (dataArray): DataArray containing % perennial snow status for filling gaps in array. 

    pSnowFree (dataArray): DataArray contianing % snow-free status for filling gaps in array.

    ax (mpl axis): Axis to plot on.

    single_plot (bool): True = Single plot, False = Subplot. 

    Returns:
    Categorical map of snow start or end over Canada. 
    """

    # Manipulate array for plotting
    if form == 'start':
        pSF_v = 1000 # Snow-free is highest value
        pP_v = -1000 # Perennial snow is lowest value
    if form == 'end':
        pSF_v = -1000 # Snow-free is lowest value
        pP_v = 1000 # Perennial snow is highest value

    array_p = xr.where(pSnowFree >= 50, pSF_v, array) # Add usually snow-free
    array_p = xr.where(pPerennial >= 50, pP_v, array_p) # Add usually perennial snow
    array_p = xr.where(array.notnull(), array, array_p) # Re-add start/end values if needed so none are lost

    # Create color scheme
    if form == 'start':
        colors = ['#4C0073', '#5E4FA2', '#4273B3', '#3F96B7', '#60BBA8', 
                '#88D0A4', '#B2E0A2', '#D7EF9B', '#EFF9A7', '#FFFFBF', 
                '#FEEC9E', '#FED480', '#FDB466', '#F88E52', '#F06744', 
                '#DD4A4C', '#C0274A', '#9E0142', '#730000']
        scale = [-1001, -999, -121.5, -111.5, -101.5, 
                -91.5, -81.5, -71.5, -60.5, -50.5, 
                -40.5, -30.5, -20.5, -10.5, -0.5, 
                10.5, 20.5, 31.5, 999, 1001]
        labels = ['Usually perennial snow', 'Before September 1', 'September 1-10', 'September 11-20', 'September 21-30', 
                'October 1-10', 'October 11-20', 'October 21-31', 'November 1-10', 'November 11-20', 
                'November 21-30', 'December 1-10', 'December 11-20', 'December 21-31', 'January 1-10', 
                'January 11-20', 'January 21-31', 'After January 31', 'Usually snow-free']    

    if form == 'end':
        colors = ['#730000', '#9E0142', '#C0274A', '#DD4A4C', '#F06744', 
                  '#F88E52', '#FDB466', '#FED480', '#FEEC9E', '#FFFFBF',
                  '#EFF9A7', '#D7EF9B', '#B2E0A2', '#88D0A4', '#60BBA8',
                  '#3F96B7', '#4273B3', '#5E4FA2', '#4C0073']
        scale = [-1001, -999, 59.5, 69.5, 79.5, 
                 90.5, 100.5, 110.5, 120.5, 130.5,
                 140.5, 151.5, 161.5, 171.5, 181.5,
                 191.5, 201.5, 212.5, 999, 1001]
        labels = ['Usually snow-free', 'Before March 1', 'March 1-10', 'March 11-20', 'March 21-31', 
                  'April 1-10', 'April 11-20', 'April 21-30', 'May 1-10', 'May 11-20', 
                  'May 21-31', 'June 1-10', 'June 10-20', 'June 21-30', 'July 1-10',
                  'July 11-20', 'July 21-31', 'After July 31', 'Usually perennial snow']

    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(scale, len(colors))     

    # Create plot
    if single_plot == True: 
        fig, ax = plt.subplots(figsize = (10, 7.5))
    ax = ax or plt.gca()    

    array_p.plot(cmap = cmap, norm = norm, ax = ax, add_labels = False, add_colorbar = False)    

    # Hide  axis ticks and labels
    ax.xaxis.set_tick_params(labelbottom = False)
    ax.yaxis.set_tick_params(labelleft = False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal') # Force square pixels 

    fig = ax.get_figure()

    # Configure colorbar
    if single_plot == True:
        patches = [mpatches.Patch(color = colors[i], label = labels[i]) for i in range(len(labels))]

        if form == 'start':
            title = 'Snow Season Start Date'
        if form == 'end':
            title = 'Snow Season End Date'

        plt.legend(handles = patches, bbox_to_anchor = (0.725, 1), loc = 2, borderaxespad = 0, fontsize = 9, title = title, 
                   title_fontsize = 10, frameon = False, labelspacing = 0.05, borderpad = 0.2)     

    #plt.show()     

    if single_plot == False:
        return fig            

######################################################################################################################################################
# Graphs #############################################################################################################################################
######################################################################################################################################################

# Creates time-series plots for the desired form.
def timeSeriesPlot(df, df_col = '', form = '', save = ''):

    """
    Parameters:
    df1 (df): Dataframe containing data to plot. Currently supports polars dataframes.  
    - Time index should be called 'time'

    df1_col (str): String of column name of interest from df1 to plot (if single column).

    form (str): Type of time-series plot to create. Currently supports: 'ims'
    - ims: IMS-based time-series of % area covered by snow/ice

    save (str): Save figure as file name (e.g., 'fig.tif'). '': Don't save. 

    Returns:
    Time-series plot of the desired form. 
    """

    if form == 'ims':

        if save == '':
            div = 1
            linewidth = 2
            markersize = 10
            fontsize = 10
            linewidth_t = 3
            y_txt = 8
            x_txtr = 3
            x_txtl = -13
        if save != '':
            div = 2.175
            linewidth = 1
            markersize = 5
            fontsize = 8
            linewidth_t = 1.5
            y_txt = 4
            x_txtr = 1
            x_txtl = -9

        # Initialize plot
        fig, ax = plt.subplots(figsize = (16 / div, 8 / div))

        # Smooth input (31 day centered median)
        df_sm = df.rolling(index_column = 'time', period = '31d', offset = '-16d').agg([pl.median(df_col)])

        # Find periods
        df_p = df_sm.with_columns(pl.when(pl.col(df_col).is_between(1, 90, closed = 'left')).then(2).otherwise(pl.col(df_col)).name.keep())
        df_p = df_p.with_columns(pl.when(pl.col(df_col) < 1).then(1).otherwise(pl.col(df_col)).name.keep())
        df_p = df_p.with_columns(pl.when(pl.col(df_col) >= 90 ).then(3).otherwise(pl.col(df_col)).name.keep()) 

        # Smooth periods (7 day centered mode) 
        df_p = df_p.rolling(index_column = 'time', period = '7d', offset = '-4d').agg(pl.col(df_col).mode().get(0))

        # Plot periods
        ax.fill_between(df['time'], -1, 101, where = (df_p[df_col] == 3), color = 'blue', edgecolor = 'blue', alpha = 0.25)
        ax.fill_between(df['time'], -1, 101, where = (df_p[df_col] == 2), color = 'purple', edgecolor = 'black', alpha = 0.25)
        ax.fill_between(df['time'], -1, 101, where = (df_p[df_col] == 1), color = 'red', edgecolor = 'black', alpha = 0.25)

        # Plot input
        plt.plot(df['time'], df[df_col], color = 'black', linewidth = linewidth) 

        dates = []

        # Find and plot annual snow minimum
        for year in np.unique(df['time'].dt.year()): # For each year

            # Filter to that year
            smin = df_sm.filter(pl.col('time').is_between(dt.date(year, 1, 1), dt.date(year, 12, 31))) 

            # Find all time-steps with minimum value and get median date
            smin = smin.with_columns(pl.when(pl.col(df_col) == smin.min()[df_col]).then(smin.min()[df_col]).name.keep()).drop_nulls().median()

            # Save date for accessing later (winter year)
            dates.append(smin['time'][0])

            # Plot marker at smoothed location
            plt.plot(smin['time'], smin[df_col], color = 'red', markeredgecolor = 'black', marker = 'o', markersize = markersize)

            # Annotate marker with date
            if smin[df_col][0] < 3: # Close to bottom, put above
                ax.annotate(smin['time'][0].strftime('%m/%d'), xy = (mdates.date2num(smin['time']), smin[df_col][0]), xytext = (-14, y_txt), 
                            textcoords = 'offset points', path_effects = [pe.withStroke(linewidth = linewidth, foreground = 'red')], 
                            fontsize = fontsize)
            if smin[df_col][0] >= 3: # Far enough from bottom, put below
                ax.annotate(smin['time'][0].strftime('%m/%d'), xy = (mdates.date2num(smin['time']), smin[df_col][0]), xytext = (-14,-y_txt-7), 
                            textcoords = 'offset points', path_effects = [pe.withStroke(linewidth = linewidth, foreground = 'red')], 
                            fontsize = fontsize)  

        # Winter year information
        for i in range(len(dates) - 1): # For the all date indexes except the last one

            # For snow start and end, first need to filter smoothed snow periods to winter year
            p_wyear = df_p.filter(pl.col('time').is_between(dt.date(dates[i].year, dates[i].month, dates[i].day),
                                                            dt.date(dates[i + 1].year, dates[i + 1].month, dates[i + 1].day),
                                                            closed = 'right')) # Day after minimum to minimum

            # No winter start/end for perennial snow (e.g., all dates transition or snow) or no snow (e.g., all dates snow free)
            if (p_wyear[df_col].max() > 1) & (p_wyear[df_col].min() == 1): # Need both non-snow and transition+ observations

                # Then find dates classified as either transition or snow
                p_wyear = p_wyear.filter(pl.col(df_col) > 1)
        
                # Winter year snow start (first day of either transition or snow period)
                sstart = p_wyear[0]['time'][0]
                plt.axvline(sstart, color = 'mediumorchid', linewidth = linewidth_t, zorder = -1)

                # Annotate line with date at middle point (to left)
                ax.annotate(sstart.strftime('%m/%d'), xy = (mdates.date2num(sstart), 50), xytext = (x_txtl, 0), textcoords = 'offset points', 
                            path_effects = [pe.withStroke(linewidth = linewidth, foreground = 'mediumorchid')], fontsize = fontsize, rotation = 90)
            
                # Winter year snow end (last day of either transition or snow period)
                send = p_wyear[-1]['time'][0]
                plt.axvline(send, color = 'mediumorchid', linewidth = linewidth_t, zorder = -1)

                # Annotate line with date at middle point (to left)
                ax.annotate(send.strftime('%m/%d'), xy = (mdates.date2num(send), 50), xytext = (x_txtr, 0), textcoords = 'offset points', 
                            path_effects = [pe.withStroke(linewidth = linewidth, foreground = 'mediumorchid')], fontsize = fontsize, rotation = -90)

        plt.ylabel('Area Covered by Snow/Ice (%)') 
        plt.margins(x = 0, y = 0)    

        plt.show()

    # Save
    if save != '':
        fig.get_figure().savefig(save, dpi = 600, bbox_inches = 'tight', pil_kwargs = {'compression': 'tiff_lzw'})

        return fig    

######################################################################################################################################################

# Creates observed vs. predicted plot, optionally with weights and quality. 
def ObsPredPlot(pred, obs, weights = [], quality = [], q_range = [], pred_label = '', obs_label = '', q_label = 'Quality', alpha = 0.5, w_size = 50): 

    """
    Parameters:
    pred (array, series etc.): Values representing x in scatterplot and x value in linear regression equation (i.e., predicted values).

    obs (array, series etc.): Values representing y in scatterplot and y value in linear regression equation (i.e., observed values).

    weights (array, series etc.): Values representing weights (0-1): impacts scatterplot point size, regression equation and stats. [] = No weights. 

    quality (array, series etc.): Values representing data pred quality: impacts scatterlot colormap. [] = No quality. 

    q_range [list of num]: Values representing range of quality to visualize in colorbar [qmin, qmax]. [] = Minimum to maxmimum. 

    pred_label (str): Label for x axis. Begins with 'Predicted '.

    obs_label (str): Label for y axis. Begins with 'Observed '.

    q_label (str): Label for quality colormap. Defaults to 'Quality'. 

    alpha (float): Transparaency of points. Between 0 (empty) and 1 (full). Default is 0.5.

    w_size (int): Controls size of weighted points. Default is 50. 
    
    Returns:
    Linear regression scatter plot. 
    """

    fig, ax = plt.subplots(figsize=(9, 9))

    # Define pmin and pmax
    pmin = min([pred.min(), obs.min()])
    pmax = max([pred.max(), obs.max()])
    buff = (pmax - pmin) / 100 # Edge buffer
    pmin = pmin - buff
    pmax = pmax + buff

    # Define qmin and qmax if not supplied
    if (len(quality) > 0) & (len(q_range) == 0):
        q_range = [quality.min(), quality.max()]

    # Linear regression
    lm = LinearRegression() # linear model
    if len(weights) == 0:
        lm.fit(pred.to_frame(), obs.to_frame()) # Fit linear model
    if len(weights) > 0:
        lm.fit(pred.to_frame(), obs.to_frame(), sample_weight = weights) # Fit linear model
    obs_pred = lm.predict(pred.to_frame()) # Predict obs using pred
    ax.plot(pred, obs_pred, color = 'k', lw = 2.5) # Plot regression line

    # Calculate metrics
    n = len(pred) # Number of samples
    res = pred - obs # Residuals (predicted - observed)

    if len(weights) == 0:
        r2 = r2_score(obs, obs_pred) # y_true = obs, y_pred = obs_pred
        acc = res.mean() # Accuracy (bias, mean of residuals)
        unc = np.average(res ** 2) ** (0.5) # Uncertainty (RMSD)
        pre = ((res - acc) ** 2).mean() ** 0.5 # Precision (bias corrected RMSD)
    if len(weights) > 0:
        r2 = r2_score(obs, obs_pred, sample_weight = weights)
        acc = np.average(res, weights = weights)
        unc = np.average(res ** 2, weights = weights) ** (0.5)
        pre = np.average((res - acc) ** 2, weights = weights) ** 0.5

    # 1-1 line
    plt.axline((pmin,pmin), (pmax, pmax), color = "gray", lw = 1, zorder = 0) # Drawn behind

    # Scatterplot
    if len(weights) == 0:
        if len(quality) == 0:
            scatter = plt.scatter(x = pred, y = obs, alpha = alpha)
        if len(quality) > 0:
            scatter = plt.scatter(x = pred, y = obs, c = quality, vmin = q_range[0], vmax = q_range[1], alpha = alpha)
    if len(weights) > 0:
        if len(quality) == 0:
            scatter = plt.scatter(x = pred, y = obs, s = weights * w_size, alpha = alpha)
        if len(quality) > 0:
            scatter = plt.scatter(x = pred, y = obs, s = weights * w_size, c = quality, vmin = q_range[0], vmax = q_range[1], alpha = alpha)

    # Regression equation
    equ_spot = (pmin + ((pmax - pmin) * 0.03), pmin + ((pmax - pmin) * 0.95))
    if lm.intercept_[0] < 0: 
        plt.annotate('y ={0: .2f}x -{1: .1f}'.format(lm.coef_[0][0], abs(lm.intercept_[0])), xy = equ_spot)
    if lm.intercept_[0] >= 0:
        plt.annotate('y ={0: .2f}x +{1: .1f}'.format(lm.coef_[0][0], lm.intercept_[0]), xy = equ_spot)

    # N
    n_spot = (pmin + ((pmax - pmin) * 0.875), pmin + ((pmax - pmin) * 0.16))
    plt.annotate(f'N = {n}', xy = n_spot)

    # R2 (observed vs predicted relationship)
    r2_spot = (pmin + ((pmax - pmin) * 0.875), pmin + ((pmax - pmin) * 0.13))
    plt.annotate(f'R$^2$ = {r2:.2f}', xy = r2_spot)

    # A (mean of predicted - observed) # Below 0 means predicted < observed
    acc_spot = (pmin + ((pmax - pmin) * 0.875), pmin + ((pmax - pmin) * 0.10))
    plt.annotate(f'A = {acc:.2f}', xy = acc_spot) # / hls.mean() * 100 For A%

    # U (RMSD)
    unc_spot = (pmin + ((pmax - pmin) * 0.875), pmin + ((pmax - pmin) * 0.07))
    plt.annotate(f'U = {unc:.2f}', xy = unc_spot) # / hls.mean() * 100 For U% 

    # P (Accuracy adjusted RMSD)
    pre_spot = (pmin + ((pmax - pmin) * 0.875), pmin + ((pmax - pmin) * 0.04))
    plt.annotate(f'P = {pre:.2f}', xy = pre_spot) # / hls.mean() * 100 For P%

    # # RMSE%
    # rmse_spot = (pmin + ((pmax - pmin) * 0.04), pmin + ((pmax - pmin) * 0.9))
    # if len(w) == 0:
    #     plt.annotate('RMSE% = {:.1f}'.format(root_mean_squared_error(x, y) / x.mean() * 100), xy = rmse_spot) # RMSE normalized by mean observed
    # if len(w) > 0:
    #     plt.annotate('RMSE% = {:.1f}'.format(root_mean_squared_error(x, y, sample_weight = w) / x.mean() * 100), xy = rmse_spot)

    # X axis
    plt.xlim(pmin, pmax)
    plt.xlabel('Predicted ' + pred_label)

    # Y axis
    plt.ylim(pmin, pmax)
    plt.ylabel('Observed ' + obs_label)

    # Legend
    if len(quality) > 0:
        cbax = fig.add_axes([0.53, 0.145, 0.25, 0.025]) # Left, Bottom, Width, Height
        cb = plt.colorbar(ScalarMappable(cmap = scatter.get_cmap(), norm = scatter.norm), cax = cbax, orientation = 'horizontal', label = q_label)
        cb.ax.xaxis.set_label_position('top')

    plt.show()
######################################################################################################################################################

# Creates residual vs. observed scatterplot overlaied by polynomials of Accuracy, Precision, Uncertainty. 
def APUcurves(pred, obs, weights = [], quality = [], q_range = [], label = '', q_label = 'Quality', alpha = 0.5, w_size = 50):

    """
    Parameters:
    pred (array, series etc.): Values representing x value in linear regression equation (i.e., predicted values). Used to calculated residuals (y). 

    obs (array, series etc.): Values representing y value in linear regression equation (i.e., observed values). On x axis in this plot. 

    weights (array, series etc.): Values representing weights: impacts scatterplot point size, regression equation and statistics. [] = No weights. 

    quality (array, series etc.): Values representing data pred quality: impacts scatterlot colormap. [] = No quality. 

    q_range [list of num]: Values representing range of quality to visualize in colorbar [qmin, qmax]. [] = Minimum to maxmimum. 

    label (str): Label for axes. X axis begins with 'Observed '. Y axis begins with 'Observed - Predicted '.

    q_label (str): Label for quality colormap. Defaults to 'Quality'. 

    alpha (float): Transparaency of points. Between 0 (empty) and 1 (full). Default is 0.5.

    w_size (int): Controls size of weighted points. Default is 50. 

    Returns:
    APU curves + scatter plot. Y axis is residuals (pred - obs), X axis is observed values.
    """

    # Define residuals for y-axis (obs is x axis)
    res = pred - obs

    fig, ax = plt.subplots(figsize=(9, 9))

    # Define qmin and qmax if not supplied
    if (len(quality) > 0) & (len(q_range) == 0):
        q_range = [quality.min(), quality.max()]

    # 0 line
    plt.axline((obs.min(), 0), (obs.max(), 0), color = "gray", lw = 1, zorder = 0) # Drawn behind

    # Scatterplot
    if len(weights) == 0:
        if len(quality) == 0:
            scatter = plt.scatter(x = obs, y = res, c = 'dimgray', alpha = alpha)
        if len(quality) > 0:
            scatter = plt.scatter(x = obs, y = res, c = quality, vmin = q_range[0], vmax = q_range[1], alpha = alpha)
    if len(weights) > 0:
        if len(quality) == 0:
            scatter = plt.scatter(x = obs, y = res, s = weights * w_size, c = 'dimgray', alpha = alpha)
        if len(quality) > 0:
            scatter = plt.scatter(x = obs, y = res, s = weights * w_size, c = quality, vmin = q_range[0], vmax = q_range[1], alpha = alpha)

    # Define APU polynomials
    poly = PolynomialFeatures(degree = 3)
    xp = poly.fit_transform(np.array(obs).reshape(-1,1))
    if len(weights) == 0:
        accSummary = sm.WLS(endog = res, exog = xp).fit().get_prediction(xp).summary_frame(alpha = 0.05) # Accuracy (bias)
    if len(weights) > 0:
        accSummary = sm.WLS(endog = res, exog = xp, weights = weights).fit().get_prediction(xp).summary_frame(alpha = 0.05)
    pre = abs(res - np.array(accSummary['mean']))
    unc = abs(res)
    if len(weights) == 0:
        preSummary = sm.WLS(endog = pre, exog = xp).fit().get_prediction(xp).summary_frame(alpha = 0.05) # Precision (cor RMSD)
        uncSummary = sm.WLS(endog = unc, exog = xp).fit().get_prediction(xp).summary_frame(alpha = 0.05) # Uncertainty (RMSD)
    if len(weights) > 0:
        preSummary = sm.WLS(endog = pre, exog = xp, weights = weights).fit().get_prediction(xp).summary_frame(alpha = 0.05)
        uncSummary = sm.WLS(endog = unc, exog = xp, weights = weights).fit().get_prediction(xp).summary_frame(alpha = 0.05) # Uncertainty (RMSD)

    # Plot APU polynomials
    order = np.argsort(obs) # For clean curves
    ax.plot(obs[order], accSummary['mean'][order], color = 'darkgreen', label = 'Accuracy')
    ax.plot(obs[order], accSummary['mean_ci_upper'][order], color = 'darkgreen',linewidth = 0.5, linestyle = '--')
    ax.plot(obs[order], accSummary['mean_ci_lower'][order], color = 'darkgreen', linewidth = 0.5, linestyle = '--')
    ax.plot(obs[order], preSummary['mean'][order], color = 'darkorange', label = 'Precision')
    ax.plot(obs[order], preSummary['mean_ci_upper'][order], color = 'darkorange', linewidth = 0.5, linestyle = '--')
    ax.plot(obs[order], preSummary['mean_ci_lower'][order],color = 'darkorange', linewidth = 0.5, linestyle = '--')
    ax.plot(obs[order], uncSummary['mean'][order], color = 'royalblue', label = 'Uncertainty')
    ax.plot(obs[order], uncSummary['mean_ci_upper'][order], color = 'royalblue', linewidth = 0.5, linestyle = '--')
    ax.plot(obs[order], uncSummary['mean_ci_lower'][order], color = 'royalblue', linewidth = 0.5, linestyle = '--')

    # X axis
    x_buff = (obs.max() - obs.min()) / 100
    plt.xlim(obs.min() - x_buff, obs.max() + x_buff)
    plt.xlabel('Observed ' + label)

    # Y axis
    y_buff = (res.max() - res.min()) / 100
    plt.ylim(min(res.min(), -res.max()) - y_buff, max(res.max(), abs(res.min())) + y_buff)
    plt.ylabel('Predicted - Observed ' + label)

    # Legend
    ax.legend(loc = 'upper right')
    if len(quality) > 0:
        cbax = fig.add_axes([0.145, 0.145, 0.25, 0.025]) # Left, Bottom, Width, Height
        cb = plt.colorbar(ScalarMappable(cmap = scatter.get_cmap(), norm = scatter.norm), cax = cbax, orientation = 'horizontal', label = q_label)
        cb.ax.xaxis.set_label_position('top')

    plt.show()

######################################################################################################################################################

# Create a correlation matrix plot from provided dataframe columns.
def corMatrix(df, save = ''):

    """
    Parameters:
    df (df): Dataframe containing columns of interest. 

    save (str): Save figure as file name (e.g., 'fig.tif'). '': Don't save. 

    Returns:
    Correlation matrix plot. 
    """

    ncol = len(df.columns) # For figure sizing

    fig = plt.figure(figsize=(ncol / 2, ncol / 2))

    plt.matshow(df.corr(), fignum = f.number, vmin = -1, vmax = 1, cmap = 'RdBu_r')

    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize = 12, rotation = 90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize = 12)

    cb = plt.colorbar(shrink = 0.8, label = 'Correlation (r)')

    cb.ax.tick_params(labelsize = 12)

    # Save
    if save != '':
        fig.get_figure().savefig(save, dpi = 600, bbox_inches = 'tight', pil_kwargs = {'compression': 'tiff_lzw'})

    plt.show()

######################################################################################################################################################

# From xarray, create stacked bar plot showing distribution of values. 
def stackedBar(das, percentage = False, schema = {}, save = ''):

    """
    Parameters:

    das (list): List of xarray dataarrays. Each dataarray should have a name and data corresponding to stacked bar in plot.
    - Currently, each da must have matching unique values. 

    percentage (bool): Whether (True) or not (False) to plot y-axis as percentage. False = pixel count. 

    schema (dict): Dictionary representing plot schema. In the form of {'Value': 'Color'}. Required. 
    - 'Value' is used to rename categorical numbers in legend. 
    - 'Color' is used for stacked bar portion colors in plot. Form: '#Color'. 

    save (str): Save figure as file name (e.g., 'fig.tif'). '': Don't save. 

    Returns:
    Stacked bar plot.
    """

    # Blank dataframe to fill
    df = pd.DataFrame(columns = ['Value'])

    for i in range(len(das)):

        # Get unique values and counts for dataarray
        uniques, counts = np.unique(das[i], return_counts = True)

        # Remove nan entries from both
        counts = counts[~np.isnan(uniques)]
        uniques = uniques[~np.isnan(uniques)]  

        # Calculate percentage
        if percentage == True:
            counts = counts * 100 / das[i].notnull().sum().item()  

        # Put into df to merge with main df     
        df['Value'] = uniques
        df[das[i].name] = counts # This works, but values need to be matching amonst das

    # Rename values for plot
    df['Value'] = list(schema.keys())

    # Transform df for plotting
    df = df.set_index('Value').T

    # Plot
    if percentage == True:
        ylabel = '%'
    if percentage == False:
        ylabel = 'Pixels'
    fig = df.plot(kind = 'bar', stacked = True, rot = 0, ylabel = ylabel, color = schema).legend(bbox_to_anchor = (1, 1))
    fig
    #plt.show()

    # Save
    if save != '':
        fig.get_figure().savefig(save, dpi = 600, bbox_inches = 'tight', pil_kwargs = {'compression': 'tiff_lzw'})

######################################################################################################################################################