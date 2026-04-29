# main_processor.py
import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np
import geopandas as gpd
from affine import Affine

from raster_input import BaseClipper, TiffClipper
from typing import Optional
class RasterMaskBase(ABC):

    @abstractmethod
    def process_mask(self, base_path: str, bound_shp: str, target_reach_id: Any, RasterInput: BaseClipper) -> Tuple[Dict[str, np.ndarray], Affine, Dict, np.ndarray]:
        pass

class DefaultMasker(RasterMaskBase):

    def process_mask(self, base_path: str, bound_shp: str, target_reach_id: Any, clipper: Optional[BaseClipper] = None) -> Tuple[Dict[str, np.ndarray], Affine, Dict, np.ndarray]:

        study_polygon = gpd.read_file(bound_shp)
        study_polygon = study_polygon[study_polygon["reach_id"] == target_reach_id]
        if study_polygon.empty:
            raise ValueError(f"No feature found with reach_id={target_reach_id} in {bound_shp}")


        if clipper is None:
            clipper = TiffClipper()

        try:
            water_arr, water_transform, meta = clipper.clip_data(base_path, "water_area", study_polygon)
        except:
            return None, None, None, None
        dist_arr, _, _ = clipper.clip_data(base_path, "cross_track", study_polygon)
        dist_arr=np.abs(dist_arr)

        if water_arr is None or dist_arr is None:
            raise RuntimeError("Failed to clip both 'water_area' and 'cross_track' data.")
        
        if water_arr.shape != dist_arr.shape:
            raise ValueError(f"Clipped 'water_area' shape {water_arr.shape} and 'cross_track' shape {dist_arr.shape} do not match.")

        candidate_mask = (
            (dist_arr >= 10000) & (dist_arr <= 60000) &
            (water_arr >= 1000) & (water_arr <= 20000)
        )
        if np.sum(candidate_mask)<10:
            return None, None, None, None
        clipped_arrays = {"water_area": water_arr, "cross_track": dist_arr}
        return clipped_arrays, water_transform, meta, candidate_mask.astype(np.uint8)
