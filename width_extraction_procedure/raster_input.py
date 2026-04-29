import os
import tempfile
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from affine import Affine
from osgeo import gdal

class BaseClipper(ABC):

    @abstractmethod
    def clip_data(self, base_file_path: str, logical_varname: str, study_polygon: gpd.GeoDataFrame) -> Tuple[np.ndarray, Affine, Dict]:

        pass
class TiffClipper(BaseClipper):

    _VARIABLE_MAPPING = {
        "water_area": {"tif_suffix": "_area.tif"},
        "cross_track": {"tif_suffix": "_dist.tif"}
    }

    def clip_data(self, base_file_path: str, logical_varname: str, study_polygon: gpd.GeoDataFrame) -> Tuple[np.ndarray, Affine, Dict]:
        if study_polygon.empty:
            raise ValueError("Provided study_polygon GeoDataFrame is empty.")

        map_info = self._VARIABLE_MAPPING.get(logical_varname)
        if not map_info:
            raise ValueError(f"Unknown logical variable name for TIF clipping: {logical_varname}")

        base_name_without_ext = os.path.splitext(base_file_path)[0]
        tif_file = base_name_without_ext + map_info["tif_suffix"]
        
        if not os.path.exists(tif_file):
            raise FileNotFoundError(f"TIF file not found for '{logical_varname}': {tif_file}")


        with rasterio.open(tif_file) as src:
            src_crs = src.crs
            study_polygon_tr = study_polygon.to_crs(src_crs)
            out_image, out_transform = mask(src, study_polygon_tr.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
        
        return out_image[0], out_transform, out_meta

class NetCDFClipper(BaseClipper):


    def clip_data(self, base_file_path: str, logical_varname: str, study_polygon: gpd.GeoDataFrame) -> Tuple[np.ndarray, Affine, Dict]:
        if study_polygon.empty:
            raise ValueError("Provided study_polygon GeoDataFrame is empty.")
        

        if not os.path.exists(base_file_path):
            raise FileNotFoundError(f"NetCDF file not found: {base_file_path}")

        tmp_tif = None
        try:
            tmp_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
            
            src_path = f"NETCDF:{base_file_path}:{logical_varname}"
            gdal.Translate(
                destName=tmp_tif,
                srcDS=src_path,
                format="GTiff",
                creationOptions=["COMPRESS=LZW"]
            )
            
            with rasterio.open(tmp_tif) as src:
                src_crs = src.crs
                study_polygon_tr = study_polygon.to_crs(src_crs)
                out_image, out_transform = mask(
                    src, study_polygon_tr.geometry, crop=True
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
            
            return out_image[0], out_transform, out_meta
        finally:
            if tmp_tif and os.path.exists(tmp_tif):
                os.remove(tmp_tif)