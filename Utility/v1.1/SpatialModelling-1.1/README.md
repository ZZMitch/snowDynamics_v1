# SpatialModelling
 Utility code supporting modelling (e.g., machine learning) and processing of geospatial data into higher level products. 

## Structure 

### Utility Code: General_Utils
Functions for general processing of geospatial data. 
#### Functions: Data Cube Processing
- *cubeZonal()*: Apply zonal statistics on a data cube time-series and output a dataframe.
#### Functions: Vector Processing
- *centerline_single()*: Centerline function that always returns a single line-string (e.g., rivers).
- *merge_mls()*: Merge a MultiLineString into a single LineString by iteratively connecting nearest endpoints.

---

### Utility Code: Snow_Utils
Functions for creating snow products. 
#### Functions: HLS (Harmonzied Landsat Sentinel-2)
- *annualFmask2SnowCube()*: Converts annual Fmask 10-cat cubes from observationAvailabilityHLS() (SpatialDataAccess > STAC_Utils) to snow cubes where snow = 1, no snow = 0, clouds etc. = NaN.
#### Functions: IMS (Interactive Multisensor Snow and Ice Mapping System)
- *createAnnualCanadaIMS()*: Converts daily northern hemisphere IMS data (as downloaded) to annual Canada IMS xarray DataArray.
- *annualIMS2SnowCube()*: Converts annual IMS 4-cat cubes from createAnnualCanadaIMS() to snow cubes where snow = 1, non-snow = 0.
- *monthlyIMSprocess()*: Load, reproject, and clip annual IMS cube in monthly increments.
#### Functions: General Snow Cube
- *cleanSnowCube()*: Clean snow cubes by identifying clear snow periods and removing outliers.
- *snowDynamics1D()*: Innter snow dynamics function that runs on 1D arrays. 
- *snowCube2SnowDynamics()*: Converts cleaned snow cube (e.g., from cleanSnowCube()) to output Dataset quantifying various snow dynamics.
- *interannualSnowDynamics()*: From snowCube2SnowDynamics() output cube, create inter-annual (multi-year) snow dynamics products.
- *mergeTiledSnowDynamics()*: Convert tiled NetCDF snow dynamics products to wide-area COG products.
---

### Utility Code: TrainingData_Utils
Functions for preparing training data.
#### Functions: Sample Locations
- *validFootprint()*: Get valid (i.e., not nodata) footprint from a raster image (e.g., high-res image from which to build training data).
- *validFootprints()*: For all images in a folder structure, get valid footprint.
- *mergeFootprints()*: Find all footprint shps in a folder structure, merge together and clip to specific AOIs. 
