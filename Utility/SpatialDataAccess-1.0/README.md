# SpatialDataAccess
Utility code supporting access (e.g., from STACs) and pre-processing (e.g., with xarray) of geospatial data. 

## Structure 

### Notebooks: 
- *buildHLS_Demo*: Generate dask-backed Harmonized Landsat Sentinel-2 (HLS) cube without duplicate observations, load into memory, process (e.g., cloud masking), visualize.
- *buildHLS_Demo_Fmask*: Same as above, but with Fmask cube instead of spectral cube.
- *buildS2_Test*: Rough test of functinality as above, with Sentinel-2 cube.

---

### Utility Code: GEE_Utils
Currently empty.

---

### Utility Code: HTTP_Utils
Functions for accessing data stored at generic HTTP links (usually by downloading). 
#### Functions: Download
- *downloadAll()*: Download all files from link that meet specified conditions.
#### Functions: Organize
- *unzipAll()*: Replace all zipped files in folder of a certain extension with unzipped files.

---

### Utility Code: PreProcess_Utils
Functions for pre-processing large datasets into desired format (e.g., HLS into clean xarray time-series). 
#### Functions: Front-End
- *xarray2GTs()*: Download all time-steps in xarray as GeoTiffs in set file system (download observations by year to limit impact of server/memory issues).
- *cleanCube()*: Clean cube by removing time-steps with high NA % OR high unclear % (from mask) and/or pixel masking based on mask.
#### Functions: Data Transfer
- *loadXR()*: Compute xarray into memory.
- *downloadNC()*: Download xarray DataArray as netCDF file for later use.
- *uploadNC()*: Upload netCDF file as xarray DataArray or DataSet.
- *downloadGT()*: Download xarray DataArray as GeoTiffs for later use (currently supports HLS/Fmask).
- *uploadGT()*: Upload GeoTiff files as xarray (e.g., outputs from downloadGT()) based on provided form. 
#### Functions: Pixel Processing
- *convertFmask()*: Convert bitpacked Fmask band to more usable formats (e.g., for cloud-masking) using hierarchy approach.
- *rescaleFill()*: Rescale cube and set fill value to NA.
- *pixelClean()*: Apply pixel level mask to cube.
#### Functions: Spatial
- *poly2bbox()*: Gets bounding box in specified projection based on input polygon.
- *latlon2bboxes()*: Gets UTM and lat/lon bounding box based on input lat/lon and edge size (m).
- *fishnet()*: Create fishnet at specified size based on input polygon.
- *subfishnet()*: Get subsquares of a specified size from a fishnet (created by fishnet()).
#### Functions: Spectral
- *bandBuilder()*: Filter bands to common set.
#### Functions: Temporal
- *removeBadScenes()*: Remove bad images from time-series cube based on metadata.
- *sameDayMerge()*: Create same-day median composites from time-series cube.
- *timestepClean()*: Clean cube by removing time-steps with high NA % OR high unclear % (from mask) and/or pixel masking based on mask.

---

### Utility Code: STAC_Utils
Functions for accessing STAC data (usually as xarray).
#### Functions: Build Cubes
- *buildHLS()*: Builds HLS spectral and/or Fmask cubes as Dask-backed Lazy Arrays.
- *buildS2()*: Builds S2 spectral and (if asked for) SCL cubes as Dask-backed Lazy Arrays.
#### Functions: Assess HLS
- *sampleMonthlyClearCountsHLS()*: Get monthly clear HLS observation counts from point data.
- *observationAvailabilityHLS()*: Get HLS observation availability and other metadata for given area(s).
- *sampleSpectralHLS()*: Placeholder.
#### Back End
- *accessSTAC()*: Access STAC catalog via link.
- *OAHLScsvs()*: Create observation and metadata csvs to fill with observationAvailabilityHLS().
- *OAHLSschema()*: Create schema for observationAvailabilityHLS() csvs.
