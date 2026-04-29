import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import norm
from scipy.ndimage import label
from scipy.spatial import cKDTree
from geopy.distance import geodesic
from rasterio.features import rasterize
from pathlib import Path
from skimage.morphology import skeletonize
import numpy as np
import rasterio
from collections import deque

from raster_input import BaseClipper,TiffClipper,NetCDFClipper
from raster_mask import RasterMaskBase, DefaultMasker
def export_candidate_mask_tif(output_path: str,
                              candidate_mask: np.ndarray,
                              transform,
                              meta: dict,dt=rasterio.uint8):
    profile = meta.copy()
    profile.update({
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "height": candidate_mask.shape[0],
        "width": candidate_mask.shape[1],
        "transform": transform,
        "compress": "LZW",
        "nodata": 0
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(candidate_mask.astype(np.uint8), 1)
def GetWidthCurve(nc_path:str,river_path:str,river_link_path:str,river_bound_path:str,target_reach_id:int,
                        RasterInput:BaseClipper=TiffClipper(),
                        RasterMasker:RasterMaskBase=DefaultMasker()):
    data_dict, water_transform, meta, candidate_mask=RasterMasker.process_mask(nc_path,river_bound_path,target_reach_id,RasterInput)
    if data_dict is None:
        return None,None
    """
        Step 1: Extract all 4-connected water pixels within the prior river area, stored in river_mask.
    """
    link_gdf = gpd.read_file(river_link_path)
    link_gdf = link_gdf[link_gdf["reach_id"] == target_reach_id]
    if link_gdf.empty:
        raise ValueError(f"No feature found with reach_id={target_reach_id} in {river_link_path}")

    link_geom = link_gdf.to_crs(meta["crs"]).geometry.union_all()

    rows, cols = np.where(candidate_mask)
    xs, ys = rasterio.transform.xy(water_transform, rows, cols, offset="center")
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs=meta["crs"])

    inside_link = points.within(link_geom)
    seeds_mask = np.zeros(candidate_mask.shape, dtype=bool)
    seeds_mask[rows[inside_link], cols[inside_link]] = True

    structure = np.array([[0, 1, 0],
                        [1, 1, 1], 
                        [0, 1, 0]], dtype=int)
    labeled, nlabels = label(candidate_mask, structure=structure)
    river_mask = np.isin(labeled, np.unique(labeled[seeds_mask]))
    # export_candidate_mask_tif(f"{nc_path.replace('.nc',f'_mask.tif')}",river_mask,water_transform,meta)
    
    """
        Step 2: For each pixel in river_mask, find the nearest SWORD point (SWORD points are spaced ~30-40 m apart)
    """
    orig_gdf = gpd.read_file(river_path)
    orig_gdf = orig_gdf[orig_gdf["reach_id"] == target_reach_id]
    if orig_gdf.empty:
        raise ValueError(f"No feature found with reach_id={target_reach_id} in {river_path}")
    orig_gdf = orig_gdf.to_crs(meta["crs"])
    line = None
    for geom in orig_gdf.geometry:
        if geom.geom_type == "LineString":
            line = geom
            break
    if line is None:
        raise ValueError("No LineString in orig_gdf")
    coords = list(line.coords)
    points_arr = np.array(coords)   # (n_points, 2)
    rows_river, cols_river = np.where(river_mask)
    xs_r, ys_r = rasterio.transform.xy(water_transform, rows_river, cols_river, offset="center")

    kdtree = cKDTree(points_arr) # KDTree
    dists, idxs = kdtree.query(np.c_[xs_r, ys_r])

    """
        Step 3: Connectivity filtering. For each pixel in river_mask, check the path to the nearest SWORD point. If it encounters non-water pixels, mark it as a potential disconnected lake. However, to avoid misclassifying pixels due to river shifts, if it encounters water pixels again after non-water pixels, we consider it connected. The final result is stored in connect_mask.
    """
    sort_idx = np.argsort(dists)
    
    step_size = abs(water_transform.a)
    rows_river, cols_river, idxs,dists = rows_river[sort_idx], cols_river[sort_idx], idxs[sort_idx],dists[sort_idx]
    xs_p, ys_p = points_arr[:, 0], points_arr[:, 1] # SWORD points
    id_invers=np.zeros_like(river_mask,dtype=int)
    for j, (r, c) in enumerate(zip(rows_river, cols_river)):
        id_invers[r,c]=j
    connect_mask = np.zeros_like(river_mask, dtype=bool)
    prior_river_mask = rasterize(
        [(link_geom, 1)],
        out_shape=river_mask.shape,
        transform=water_transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)
    angles = np.deg2rad(np.linspace(-25, 25, 11))  # ±25°
    rot_mats = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angles]
    max_steps=100 # width < 10000 m
    for r, c, idx in zip(rows_river, cols_river, idxs):
        assert river_mask[r, c], f"Unexpected non-water pixel at {(r, c)}"

        # in prior mask → retain
        if prior_river_mask[r, c]:
            connect_mask[r, c] = True
            continue
        cx, cy = xs_p[idx], ys_p[idx]  # SWORD point coordinates
        x, y = rasterio.transform.xy(water_transform, r, c, offset="center")  # raster center
        dir_vec = np.array([cx - x, cy - y], dtype=float) # shortest path direction
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            continue # what happened ?
        dir_vec /= norm

        success = False
        for rot in rot_mats: # at most 25°
            dvec = rot @ dir_vec  # rotation matrix
            dvec /= np.linalg.norm(dvec)
            meet_non_water=False
            into_prior_mask=False
            for step in range(1, max_steps + 1):
                # 100 m every step
                x_new = x + dvec[0] * step * step_size
                y_new = y + dvec[1] * step * step_size
                rr, cc = rasterio.transform.rowcol(water_transform, x_new, y_new)
                if rr < 0 or rr >= river_mask.shape[0] or cc < 0 or cc >= river_mask.shape[1]:
                    break # walk too far
                if not connect_mask[rr, cc]:
                    meet_non_water=True
                if prior_river_mask[rr, cc]:
                    into_prior_mask=True
                    if connect_mask[rr, cc]:
                        # ok we reach water pixel
                        if not meet_non_water:
                            success=True
                        break
                else:
                    if into_prior_mask:
                        # U go out of boundary but never meet water pixel, consider it disconnected
                        # Still ok
                        success = True
                        break
            if success:
                break
        if success:
            connect_mask[r, c] = True
    connect_mask=connect_mask.astype(np.uint8)
    # export_candidate_mask_tif(f"{nc_path.replace('.nc',f'_connect_mask.tif')}",connect_mask,water_transform,meta)
    print(f"Finish direction filtering: {connect_mask.sum()} / {river_mask.sum()}")
    if np.sum(connect_mask)<10:
        return None,None

    """
        Step 4: Skeletonization filtering. We skeletonize the connect_mask to get skeleton_mask.

    """
    skeleton_mask = skeletonize(connect_mask > 0)
    # export_candidate_mask_tif(f"{nc_path.replace('.nc',f'_skeleton.tif')}",skeleton_mask.astype(np.uint8),water_transform,meta)
    sword2skeleton_min=np.full_like(xs_p,np.inf,dtype=float)
    for r, c, idx, dist_m in zip(rows_river, cols_river, idxs, dists):
        if not skeleton_mask[r,c]:
            continue
        sword2skeleton_min[idx]=min(sword2skeleton_min[idx],dist_m)
    reach_len=link_gdf["reach_len"].iloc[0]
    reach_itv=reach_len/len(xs_p)
    range_smooth=int(200//reach_itv)+1 # 200 m smoothing range, in number of points
    sk_min=np.zeros_like(sword2skeleton_min,dtype=float)
    for i in range(len(xs_p)):
        left_id=max(0,i-range_smooth)
        right_id=min(len(xs_p),i+range_smooth+1)
        sk_min[i]=np.min(sword2skeleton_min[left_id:right_id])
    skeleton_core = np.zeros_like(river_mask, dtype=bool)
    id_core=[]
    for j, (r, c, idx, dist_m) in enumerate(zip(rows_river, cols_river, idxs, dists)):
        if skeleton_mask[r,c] and dist_m<=100+sk_min[idx]:
            skeleton_core[r,c]=True # Strict, so may not cover all skeleton pixels stand for centerline, but can avoid some wrong pixels
            id_core.append(j)
    # export_candidate_mask_tif(nc_path.replace('.nc', '_core_skeleton.tif'), skeleton_core, water_transform, meta)
    bfs_dist=np.zeros((len(id_core),len(rows_river)),dtype=int)
    bfs_mark=np.zeros_like(river_mask, dtype=int)
    dirs8 = [(-1,-1), (-1,0), (-1,1),
            ( 0,-1),          ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)]
    for j in range(len(id_core)):
        start_id=id_core[j]
        
        q = deque()
        q.append(start_id)
        r0, c0 = rows_river[start_id], cols_river[start_id]
        bfs_mark[r0, c0] = j + 1

        while q:
            cur_id = q.popleft()
            r,c=rows_river[cur_id], cols_river[cur_id]
            new_d=bfs_dist[j,cur_id]+1
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= river_mask.shape[0] or nc >= river_mask.shape[1]:
                    continue
                if not skeleton_mask[nr,nc]:
                    continue
                if bfs_mark[nr, nc] == j + 1:
                    continue
                idx_next=id_invers[nr,nc]
                bfs_dist[j, idx_next] = new_d
                bfs_mark[nr, nc] = j + 1
                q.append(idx_next)
    skeleton_final=np.zeros_like(river_mask, dtype=bool)
    for i in range(len(rows_river)):
        r,c=rows_river[i], cols_river[i]
        if not skeleton_mask[r,c]:
            continue
        ok_flag=False
        if skeleton_core[r,c]:
            ok_flag=True
        for j1 in range(len(id_core)):
            if ok_flag:
                break
            j1_id=id_core[j1]
            for j2 in range(j1):
                # wether j1->i->j2 is shortest path
                if bfs_dist[j2][j1_id]==bfs_dist[j2][i]+bfs_dist[j1][i]:
                    ok_flag=True
                    break
        if ok_flag:
            skeleton_final[r,c]=True
    # export_candidate_mask_tif(nc_path.replace('.nc', '_optm_skeleton.tif'), skeleton_final, water_transform, meta)
    """
        Step 5: Get final mask. We BFS from skeleton_final pixels. If it encounters skeleton pixels, we retain it; if it encounters non-skeleton pixels, we check whether it is in the prior river mask. If it is in the prior river mask, we also retain it; otherwise, we discard it. The final result is stored in final_mask.
    """
    q = deque()
    for j, (r, c) in enumerate(zip(rows_river, cols_river)):
        if skeleton_final[r,c]:
            q.append((j,1))
            bfs_mark[r,c]=-1
    for j, (r, c) in enumerate(zip(rows_river, cols_river)):
        if skeleton_mask[r,c] and not skeleton_final[r,c]:
            q.append((j,-1))
            bfs_mark[r,c]=-1
    final_mask= np.zeros_like(river_mask, dtype=bool)
    final_ids=[]
    while q:
        cur_id,typ = q.popleft()
        r,c=rows_river[cur_id], cols_river[cur_id]
        if typ==1 or prior_river_mask[r,c]:
            final_mask[r,c]=True
            final_ids.append(cur_id)
        for dr, dc in dirs8:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= river_mask.shape[0] or nc >= river_mask.shape[1]:
                continue
            if not connect_mask[nr,nc]:
                continue
            if bfs_mark[nr, nc] == -1:
                continue
            idx_next=id_invers[nr,nc]
            bfs_mark[nr, nc] = -1
            q.append((idx_next,typ))
    print(f"Skeleton :{final_mask.sum()} / {river_mask.sum()}")
    if np.sum(final_mask)<10:
        return None, None
    export_candidate_mask_tif(nc_path.replace('.nc', f'_final_mask_{target_reach_id}.tif'), final_mask, water_transform, meta)
    final_ids=np.array(final_ids)
    rows_river_final=rows_river[final_ids]
    cols_river_final=cols_river[final_ids]
    idxs_final=idxs[final_ids]
    river_vals = np.minimum(data_dict['water_area'][rows_river_final, cols_river_final], 12000)
    xtrk_dist=np.nanmedian(data_dict['cross_track'][rows_river_final, cols_river_final])
    sums_per_vertex = np.bincount(idxs_final, weights=river_vals, minlength=len(points_arr)).astype(float)

    sums_list = sums_per_vertex.tolist()
    return sums_list,xtrk_dist



def CalculateDistance(point1, point2):
    return geodesic(point1, point2).meters
def ReachPrior(river_path:str, target_reach_id, limit_distance=500):
    gdf = gpd.read_file(river_path)
    reach = gdf[gdf['reach_id'] == target_reach_id]
    
    if not reach.empty:
        geometry = reach.geometry.iloc[0]
        
        points = list(geometry.coords)
        points = [(lat, lon) for lon, lat in points]
        cumulative_distances = [0,]
        total_distance = 0
        
        for i in range(1, len(points)):
            dist = CalculateDistance(points[i-1], points[i])
            total_distance += dist
            cumulative_distances.append(total_distance)
        cumulative_distances=np.array(cumulative_distances)*total_distance/cumulative_distances[-1]
        rt_list=[]
        for i in range(0, len(cumulative_distances)):
            start_idx=np.where(cumulative_distances>=cumulative_distances[i]-limit_distance)[0][0]
            end_idx=np.where(cumulative_distances<=cumulative_distances[i]+limit_distance)[0][-1]
            dist_straight=CalculateDistance(points[start_idx],points[end_idx])
            dist_path=cumulative_distances[end_idx]-cumulative_distances[start_idx]
            rt_list.append(dist_straight/dist_path)

        results = {
            'distance': cumulative_distances,
            'ratio': rt_list
        }
        return results
    else:
        raise ValueError(f"No reach found with reach_id {target_reach_id}")



def gaussian_smooth(dist: np.ndarray, w_raw: np.ndarray, window_size: int = 500, sigma: float = 250) -> np.ndarray:

    w_smooth = np.zeros_like(w_raw)

    for i in range(3,len(dist)-3):
        center_distance = dist[i]
        
        idx_left=np.where(dist >= center_distance - window_size)[0][0]
        idx_right=np.where(dist <= center_distance + window_size)[0][-1]
    
        dist_within_window = dist[idx_left:idx_right+1]
        w_within_window = w_raw[idx_left:idx_right+1]

        gauss_weights = norm.pdf(dist_within_window, loc=center_distance, scale=sigma)
        
        gauss_weights /= gauss_weights.sum()
        
        w_smooth[i] = np.sum(gauss_weights * w_within_window)

    return w_smooth
def GetReachCurve(nc_folder:str,river_path:str,river_link_path:str,river_bound_path:str,target_reach_id:int):
    prior_info=ReachPrior(river_path,target_reach_id)
    dist=np.array(prior_info['distance'])
    nR=len(dist)
    nc_base = Path(nc_folder).resolve()
    filenames=[]
    curves=[]
    xtrks=[]
    for nc_file in nc_base.glob("*.nc"):
        cur_nc_path = str(nc_file)
        print(f"Processing: {cur_nc_path}")
        w_raw,xtrk_dist=GetWidthCurve(cur_nc_path,river_path,river_link_path,river_bound_path,target_reach_id)
        if w_raw is None:
            continue
        w_raw=np.array(w_raw)
        w_raw[:3]=0
        w_raw[-3:]=0
        for i in range(3,len(w_raw)-3):
            w_raw[i]/=0.5*(dist[i+1]-dist[i-1])
        w_smoothed=gaussian_smooth(dist,w_raw,500,250)
        filenames.append(Path(cur_nc_path).stem)
        curves.append(w_smoothed)
        xtrks.append(xtrk_dist)
    prior_info['distance']=dist
    prior_info['filename']=filenames
    prior_info['width_curve']=curves
    prior_info['xtrk']=xtrks
    return prior_info