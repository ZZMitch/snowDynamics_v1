# SpatialVisualization
 Utility code supporting visualization of geospatial data. 

## Structure 

### Utility Code: Viz_Utils
Functions for visualizing data (e.g., with matplotlib, hvplot and similar Python tools). 
#### Functions: Maps
- *matrixPlot()*: Creates a plot matrix from xarray data cube.
- *adjustTime()*: Adjusts time coordinates of xarray cube for plotting, including adding satellite.
- *mapCanada()*: Map a single variable from a raster (e.g., dataArray) or vector (gdf, e.g., tiles) over Canada.
- *mapCanadaSnowStartEnd()*: ap Canada-wide snow season start and end dates as easy-to-interpret categories.
#### Functions: Graphs
- *timeSeriesPlot()*: Creates time-series plots for the desired form.
- *obsPredPlot()*: Creates observed vs. predicted plot, optionally with weights and quality.
- *APUcurves()*: Creates residual vs. observed scatterplot overlaied by polynomials of Accuracy, Precision, Uncertainty. 
- *corMatrix()*: Create a correlation matrix plot from provided dataframe columns.
- *stackedBar()*: From xarray, create stacked bar plot showing distribution of values. 

---
