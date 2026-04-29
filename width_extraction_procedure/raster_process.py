from raster_prep import GetReachCurve
from auxi import Filename2Numbers, Filename2Datetime, Dist2Q
from width_smoother import HampelFilter, GetSmoother
import os
import pickle
import numpy as np
import geopandas as gpd
from collections import defaultdict
def CalculateWidth(nc_base:str,original_shp:str,link_shp:str,bound_shp:str,
        target_reach_id:int,save_path:str,debug_mode=False):
    """
    nc_base: the directory where the raster files are located
    original_shp: the original shapefile containing the reach geometry
    link_shp: the strict boundary shapefile
    bound_shp: the loose (clipped) boundary shapefile
    target_reach_id: the ID of the target reach
    save_path: the path to save/load the processed results
    debug_mode: whether to print debug information
    """
    # Step 1: Obtain optimized mask for raster
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results=GetReachCurve(nc_base,original_shp,link_shp,bound_shp,target_reach_id)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
    target_gdf = gpd.read_file(original_shp)
    target_shapes = target_gdf[target_gdf['reach_id'] == target_reach_id]
    target_shape = target_shapes.iloc[0]
    width_p = target_shape['width']
    length_p = target_shape['reach_len']
    if debug_mode:
        print(f"width: {width_p},length: {length_p}")
    # Step 2: Process by pass, because some tracks have systematic biases or no values in certain areas
    pass_dict=defaultdict(list)
    width_limit=max(20,0.2*width_p) # Minimum width
    zero_pos=defaultdict(list)
    for filename, curve in zip(results['filename'], results['width_curve']):
        if np.nanmax(curve)<width_limit:
            continue
        _,pas = Filename2Numbers(filename)
        pass_dict[pas].append(curve)
        none_pos=np.where(curve<1)[0]
        if len(none_pos)>0.1*len(curve):
            # Mean that the raster does not fully cover the river reach
            half_len=len(curve)//2
            left_more=2*np.sum(none_pos<half_len)-len(none_pos)
            # >0 : more on the left, <0: more on the right, =0: balanced
            if left_more>0:
                zero_p=none_pos[-1]
            else:
                zero_p=none_pos[0]
            zero_pos[pas].append(zero_p)
    if debug_mode:
        for pas,widths in pass_dict.items():
            print(pas,np.median(widths))
    track_median={}
    track_nstd={}
    # track=pass
    # This calculates the aggregated results for each track at the spatial scale, so the result is a spatial-scale curve
    for pas_id, curves in sorted(pass_dict.items()):
        curves_array = np.array(curves) # shape: (n_curves, curve_length)
        median_curve = np.median(curves_array, axis=0)
        nstd_curve = np.std(curves_array, axis=0)/(median_curve+1)
        nstd_curve[nstd_curve>3]=3 # >3 means that the curve is very unstable, usually due to outliers, so we set a cap to avoid too much influence
        track_median[pas_id]=median_curve
        track_nstd[pas_id]=nstd_curve
    # Step 3: Plan to select a sub-reach of length 3 km, enumerate each selection, and consider multiple indicators
    window_length = 1500  # 3 km
    dist=np.array(results['distance'])
    pass_ids = list(track_median.keys())
    consistency = np.zeros_like(dist) # Consistency of different tracks' median curves in terms of spatial variation trend, higher is better
    max_nstd=np.zeros_like(dist) # Indicates whether there are significant outliers, too large is not good
    mean_wind=np.zeros_like(dist) # Average winding degree, ideal is 1
    windingness=np.array(results['ratio']) # Calculation method is the straight-line distance/window range distance, larger is better
    for i in range(len(dist)):
        start =np.where(dist>=dist[i]-window_length)[0][0]
        end = np.where(dist<=dist[i]+window_length)[0][-1]
        window_median = np.array([track_median[pass_id][start:end] for pass_id in pass_ids])
        window_nstd=np.array([track_nstd[pass_id][start:end] for pass_id in pass_ids])
        max_nstd[i]=np.nanmax(window_nstd)
        mean_wind[i]=np.nanmean(windingness[start:end])
        if window_median.shape[0] > 1:
            correlation_matrix = np.corrcoef(window_median)
            consistency[i] = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, 1)]))
            if np.isnan(consistency[i]): # This means one track is all zeros, usually because a track is all zeros, which means it's not usable
                consistency[i]=0.4
        else:
            consistency[i] = 1
        if consistency[i]<=0:
            continue # This is too bad
        for j in range(len(window_median)):
            if np.nanmin(window_median[j])<width_limit:
                consistency[i]*=0.5 # Best if all tracks have values
    # Quantitative calculation of score, select the best
    nstd_limit=max(0.5,np.quantile(max_nstd,0.2)) # Should not be too large
    rest_cc=consistency[max_nstd<=nstd_limit] # Prevent the last one from being left out
    cc_limit=min(0.5,np.quantile(rest_cc, 0.8)) # Should not be too small
    score=0.5*mean_wind**2+0.3*consistency-0.2*max_nstd # Select the highest score
    bad_mask=(max_nstd>nstd_limit)|(consistency<cc_limit)|(dist<window_length)|(dist>dist[-1]-window_length)
    for pass_id in zero_pos.keys():
        pos=int(np.nanmedian(zero_pos[pass_id]))
        # print('? ',pass_id,pos,dist[pos])
        # At point pos, break the connection
        infl_mask=np.abs(dist-dist[pos])<window_length # Too close and affected
        infl_val=np.exp(-2 * ((dist-dist[pos]) / window_length)**2) # Maximum deduction is 1, minimum is 18%
        score[infl_mask]-=infl_val[infl_mask]
    score[bad_mask]=-1e5
    optm_idx=np.argmax(score)
    optm_start=np.where(dist>=dist[optm_idx]-window_length)[0][0]
    optm_end=np.where(dist<=dist[optm_idx]+window_length)[0][-1]
    if debug_mode:
        print(optm_idx,dist[optm_idx])
        print(f"Range: {dist[optm_start]}-{dist[optm_end]}")
    widths={}
    widths['id_start']=optm_start
    widths['id_end']=optm_end
    
    cycle_list=[]
    pass_list=[]
    time_list=[]
    width_list=[]
    width_u_list=[]
    xtrk_list=[]
    for filename, curve, xtrk in zip(results['filename'], results['width_curve'], results['xtrk']):
        if np.max(curve)<width_limit:
            continue
        sub_curve=np.array(curve)[optm_start:optm_end+1]
        w=np.nanmean(sub_curve)
        if w<width_limit:
            continue
        cycle,pas = Filename2Numbers(filename)
        time_list.append(Filename2Datetime(filename))
        cycle_list.append(cycle)
        pass_list.append(pas)
        width_list.append(w)
        width_u_list.append(np.nanstd(sub_curve))
        xtrk_list.append(xtrk)
    time_list=np.array(time_list)
    time_idx=np.argsort(time_list)
    time_list=time_list[time_idx]
    cycle_list=np.array(cycle_list)[time_idx]
    pass_list=np.array(pass_list)[time_idx]
    width_list=np.array(width_list)[time_idx]
    width_u_list=np.array(width_u_list)[time_idx]
    xtrk_list=np.array(xtrk_list)[time_idx]
    multi_mask=np.zeros_like(time_list,dtype=bool)
    
    for i in range(1,len(multi_mask)):
        if (time_list[i]-time_list[i-1]).seconds<3600:
            multi_mask[i-1]=True
    cycle_list=cycle_list[~multi_mask]
    pass_list=pass_list[~multi_mask]
    time_list=time_list[~multi_mask]
    width_list=width_list[~multi_mask]
    width_u_list=width_u_list[~multi_mask]
    xtrk_list=xtrk_list[~multi_mask]
    widths['width_r']=width_list
    widths['width_r_u']=width_u_list
    ### Post processing
    # # Post 1
    # w1=np.zeros_like(width_list)
    # unique_pass=np.unique(pass_list)
    # w_in_pass=[]
    # q_in_pass=[]
    # for pas in unique_pass:
    #     pass_mask=(pass_list == pas)
    #     w=np.median(width_list[pass_mask])
    #     xtrks=np.median(xtrk_list[pass_mask])
    #     w_in_pass.append(w)
    #     q_in_pass.append(Dist2Q(xtrks))
    # w_in_pass=np.array(w_in_pass)
    # q_in_pass=np.array(q_in_pass)
    # width_avg=np.sum(w_in_pass*q_in_pass)/np.sum(q_in_pass)
    # if debug_mode:
    #     print(width_avg)
    # for pas,w in zip(unique_pass,w_in_pass):
    #     pass_mask=(pass_list == pas)
    #     cali_rate=width_avg/w
    #     w1[pass_mask]=width_list[pass_mask]*cali_rate
    #     if debug_mode:
    #         print(f"{pas}: mutiply {cali_rate}")
    # Post 2: Hampel filter
    w1=width_list.copy()
    w2=HampelFilter(w1)
    # Post 3: Gauss smoother, bandwidth 10 days
    times=time_list-time_list[0]
    times=np.array([ti.total_seconds()/86400 for ti in times]) # 天数
    w3=GetSmoother('gaussian',h=10).smooth(times,w2)
    widths['cycle']=cycle_list
    widths['pass']=pass_list
    widths['time']=time_list
    widths['width']=w3
    return widths
def GetReachWidth(nc_base,original_shp,link_shp,bound_shp,target_reach_id,
                curve_folder,data_folder,debug_mode=False):
    curve_file=os.path.join(curve_folder,f"reach_{target_reach_id}.pkl")
    data_file=os.path.join(data_folder,f"reach_{target_reach_id}.pkl")
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            widths = pickle.load(f)
    else:
        widths=CalculateWidth(nc_base,original_shp,link_shp,bound_shp,target_reach_id,
            curve_file,debug_mode)
        with open(data_file, 'wb') as f:
            pickle.dump(widths, f)
    return widths