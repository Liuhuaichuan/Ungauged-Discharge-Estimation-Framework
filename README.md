# An Ungauged Discharge Estimation Framework Using SWOT Observations

This repository provides an open-source implementation of a framework for estimating river discharge in ungauged basins using SWOT (Surface Water and Ocean Topography) satellite observations. The repository includes core codes, example data, and workflow demonstrations.

---

##  Repository Structure

- **`src/`**  
  Contains the core modules and functions of the framework.
  
- **`width_extraction_procedure/`**  
  Contains the core codes of the automated width extraction procedure using the SWOT Raster Product.

- **`sample_reach/`**  
  Provides a worked example for reach **73254400101**, demonstrating the discharge estimation process.

- **`sample_excel/`**  
  Contains optimized river width data for the same reach, derived from raster files.

---

## Running / Usage notes

- Before running any Jupyter notebooks or Python scripts, **adjust file paths as instructed in the comments** found at the top of each notebook/script.  
- The relative position between the Jupyter notebooks and the `src/` folder **must remain as in this repository**. Start your Jupyter server with the repository root as the working directory so imports like `from src...` work correctly.  

## References

1. **Fang et al. (2025). Improved water level retrieval in complex riverine environments: Sentinel-3 and Sentinel-6 altimetry over China's rivers.**  
   DOI: https://doi.org/10.1029/2024WR039705
   
   GitHub: https://github.com/Fangchq/An-improved-waveform-retracking-method/tree/master

3. **Tuozzolo et al. (2019). Estimating River Discharge With Swath Altimetry: A Proof of Concept Using AirSWOT Observations.**  
   DOI: https://doi.org/10.1029/2018GL080771
