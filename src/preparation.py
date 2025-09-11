import os
import time
import datetime
import requests
import numpy as np
import pandas as pd
from io import StringIO
def identify_poor_condition(df_reach):
    df_reach['poor_condit'] = None
    for i, row in df_reach.iterrows():
        q_bitwise=row['reach_q_b']
        positions = []
        binary=bin(q_bitwise)[2:]
        for position in range(len(binary)):
            if binary[len(binary)-position-1]=='1':
                positions.append(position+1)

        df_reach.at[df_reach.index[i], 'poor_condit'] = positions
        #print(positions)
    return df_reach

def time_process(df_reach):
    df_reach['standard_time']=pd.to_datetime(df_reach['time_str']).dt.tz_localize(None)
    dates=[]
    for i, item in df_reach.iterrows():
        date_only = item['standard_time'].date()
        # Convert back to datetime with time set to 00:00:00
        dates.append(datetime.datetime.combine(date_only, datetime.time.min))

    df_reach['date']=dates
    return df_reach

def remove_outlier(df_reach, prior_dem):
    """
    fluctuation=p90-p10
    remain [dem-fluctuation,dem+1.8*fluctuation]
    """
    df_wses=np.array(df_reach['wse'].tolist())
    fluctuation=np.percentile(df_wses,90)-np.percentile(df_wses,10)
    df_reach=df_reach[(df_reach['wse']>prior_dem-fluctuation) & (df_reach['wse']<prior_dem+1.8*fluctuation)]
    #df_reach=df_reach.reset_index()
    return df_reach

def merge_neighbor(df_reach,pre_common_dates, average_cloumns=['wse'],special_cloumns=['slope2'], time_name='standard_time'):
    """
    Merge data in 2 days
    """
    result_df = df_reach.copy()
    result_df = result_df.sort_values(by=time_name)
    i = 0
    while i < len(result_df) - 1:
        time_diff = (result_df.iloc[i+1][time_name] - result_df.iloc[i][time_name]).days
        if time_diff < 1:
            for column in average_cloumns:
                result_df.at[result_df.index[i], column] = (result_df.iloc[i][column]+ result_df.iloc[i+1][column])/2
            for column in special_cloumns:
                slice_data = result_df.iloc[i:i+2][column]
                positive_values = slice_data[slice_data > 0]
                if len(positive_values) > 0:
                    result_df.at[result_df.index[i], column] = positive_values.mean()
                else:
                    result_df.at[result_df.index[i], column] = 0
            result_df.at[result_df.index[i], 'reach_q']=np.min([result_df.iloc[i]['reach_q'], result_df.iloc[i+1]['reach_q']])
            result_df.at[result_df.index[i], 'poor_condit']=np.intersect1d(result_df.iloc[i]['poor_condit'], result_df.iloc[i+1]['poor_condit']).tolist()

            if result_df.iloc[i]['date'] in pre_common_dates:
                pass
            elif result_df.iloc[i+1]['date'] in pre_common_dates:
                result_df.at[result_df.index[i], 'date']=result_df.at[result_df.index[i+1], 'date']
                result_df.at[result_df.index[i], 'time_str']=result_df.at[result_df.index[i+1], 'time_str']
                result_df.at[result_df.index[i], 'standard_time']=result_df.at[result_df.index[i+1], 'standard_time']
            else:
                pass
                
            result_df=result_df.drop(result_df.index[i+1])
        else:
            i+=1
    return result_df
def get_raster_width(reachid,time_str_list,raster_path_base,width_sub):
    # return raster width list
    if width_sub=='optm':
        sub='raw'
    else:
        sub='sft'
    file_path=os.path.join(raster_path_base,f'width_{sub}_{reachid}.xlsx')
    print(file_path)
    if not os.path.exists(file_path):
        print(f"{reachid} raster not exists")
        return None
    df_raster=pd.read_excel(file_path)
    ras_times=np.array(df_raster['time'].tolist())
    ras_widths=np.array(df_raster['width'].tolist())
    result=[]
    for time_str in time_str_list:
        if time_str=='no_data':
            result.append(-1)
            continue
        time_date=datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
        swot_base=datetime.datetime(2000,1,1)
        time_sec=(time_date-swot_base).total_seconds()
        match_width=-1
        for ras_time,ras_width in zip(ras_times,ras_widths):
            time_diff=np.abs(time_sec-ras_time)
            if time_diff<3600:
                # choose the latter one
                match_width=ras_width
        result.append(match_width)
    print(f"reach {reachid} width optimized")
    return np.array(result)
reach_df_dict={}
def get_pre_common_dates(target_reachid, prior_dem,width_sub,raster_path_base, slope_used='slope2', maxrequest=10):
    import time
    #by reach
    attempt_count = 0

    while attempt_count < maxrequest:
        try:
            saved_flag=(target_reachid in reach_df_dict)
            if not saved_flag:
                hydrocron_response = requests.get(
                    f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?collection_name=SWOT_L2_HR_RiverSP_2.0&feature=Reach&feature_id={target_reachid}"
                    f"&start_time=2023-01-01T00:00:00Z&end_time=2025-03-20T00:00:00Z&output=csv&fields=p_lon,p_lat,reach_id,time_str,wse,width,{slope_used},reach_q,reach_q_b,geometry"
                ).json()
            if saved_flag or ('results' in hydrocron_response and 'csv' in hydrocron_response['results']):
                if saved_flag:
                    csv_str=reach_df_dict[target_reachid]
                else:
                    csv_str=hydrocron_response['results']['csv']
                    reach_df_dict[target_reachid]=csv_str
                df_reach = pd.read_csv(StringIO(csv_str))
                df_reach=remove_outlier(df_reach, prior_dem)
                
                if width_sub!='orig':
                    raster_width=get_raster_width(target_reachid,df_reach['time_str'].tolist(),raster_path_base,width_sub)
                    #print(raster_width)
                    if raster_width is not None:
                        df_reach['width']=raster_width
                        
                df_reach=df_reach[(df_reach['time_str']!='no_data')&(df_reach['wse']>-100)&(df_reach['width']>0)].reset_index()
                #time process
                remove_columns=['p_lon_units','p_lat_units','wse_units',f'{slope_used}_units','width_units']
                df_reach=time_process(df_reach)
                df_reach=df_reach.drop(columns=remove_columns)
                
                # df_reach=df_reach[df_reach[slope_used]>0]
                break
            else:
                attempt_count += 1
                time.sleep(1)
                #continue
        except Exception as e:
            print(f"Error occurred: {e}")
            attempt_count += 1
            time.sleep(1)

    if attempt_count==maxrequest:
        print('Exceeded maximum number of requests, unable to obtain target reach observations')
        return None
    return df_reach
def get_preprocessed_df(target_reachid,pre_common_dates, prior_dem,width_sub,raster_path_base, slope_used='slope2', maxrequest=10):
    import time
    #by reach
    attempt_count = 0

    while attempt_count < maxrequest:
        try:
            saved_flag=(target_reachid in reach_df_dict)
            if not saved_flag:
                hydrocron_response = requests.get(
                    f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?collection_name=SWOT_L2_HR_RiverSP_2.0&feature=Reach&feature_id={target_reachid}"
                    f"&start_time=2023-01-01T00:00:00Z&end_time=2025-03-20T00:00:00Z&output=csv&fields=p_lon,p_lat,reach_id,time_str,wse,width,{slope_used},reach_q,reach_q_b,geometry"
                ).json()
            if saved_flag or ('results' in hydrocron_response and 'csv' in hydrocron_response['results']):
                if saved_flag:
                    csv_str=reach_df_dict[target_reachid]
                else:
                    csv_str=hydrocron_response['results']['csv']
                    reach_df_dict[target_reachid]=csv_str
                df_reach = pd.read_csv(StringIO(csv_str))
                df_reach=remove_outlier(df_reach, prior_dem)
                
                if width_sub!='orig':
                    raster_width=get_raster_width(target_reachid,df_reach['time_str'].tolist(),raster_path_base,width_sub)
                    #print(raster_width)
                    if raster_width is not None:
                        df_reach['width']=raster_width
                        
                df_reach=df_reach[(df_reach['time_str']!='no_data')&(df_reach['wse']>-100)&(df_reach['width']>0)].reset_index()
                #time process
                remove_columns=['p_lon_units','p_lat_units','wse_units',f'{slope_used}_units','width_units']
                df_reach=time_process(df_reach)
                df_reach=df_reach.drop(columns=remove_columns)
                df_reach=identify_poor_condition(df_reach)
                
                df_reach=merge_neighbor(df_reach,pre_common_dates, average_cloumns=['wse','width'],special_cloumns=[slope_used], time_name='standard_time')
                # df_reach=df_reach[df_reach[slope_used]>0]
                break
            else:
                attempt_count += 1
                time.sleep(1)
                #continue
        except Exception as e:
            print(f"Error occurred: {e}")
            attempt_count += 1
            time.sleep(1)

    if attempt_count==maxrequest:
        print('Exceeded maximum number of requests, unable to obtain target reach observations')
        return None
    return df_reach
def reach_group_filter(df_reaches_test):
    df_reaches=df_reaches_test
    fluc_list=[]
    for df in df_reaches:
        wse=np.array(df['wse'].tolist())
        fluc_list.append(np.percentile(wse,90)-np.percentile(wse,10))
    print('fluctuation:',fluc_list)
    check_dict={}
    while 1:
        cur_df_id=None
        cur_idx=None
        max_fluc=-1
        cur_time=None
        cur_wse=None
        for i in range(len(df_reaches)):
            # find the max fluctuation in the group
            time=np.array(df_reaches[i]['standard_time'].tolist())
            wse=np.array(df_reaches[i]['wse'].tolist())
            for j in range(len(time)):
                if (i,j) in check_dict:
                    continue
                fluc=0
                f_cnt=0
                if j!=0:
                    fluc+=np.abs(wse[j]-wse[j-1])
                    f_cnt+=1
                if j!=len(time)-1:
                    fluc+=np.abs(wse[j]-wse[j+1])
                    f_cnt+=1
                if f_cnt==0:
                    continue
                fluc/=f_cnt
                if fluc<0.5*fluc_list[i]:
                    # too small change
                    continue
                if fluc>max_fluc:
                    max_fluc=fluc
                    cur_df_id=i
                    cur_idx=j
                    cur_time=time[j]
                    cur_wse=wse[j]
        if cur_df_id is None:
            break
        check_dict[(cur_df_id,cur_idx)]=1
        assert cur_idx<len(df_reaches[cur_df_id])
        long_start=cur_time-datetime.timedelta(days=22)
        long_end=cur_time+datetime.timedelta(days=22)
        short_start=cur_time-datetime.timedelta(hours=12)
        short_end=cur_time+datetime.timedelta(hours=12)
        other_diff=[]
        check_diff=None
        for i in range(len(df_reaches)):
            time=np.array(df_reaches[i]['standard_time'].tolist())
            wse=np.array(df_reaches[i]['wse'].tolist())
            long_idx=np.where((time>=long_start)&(time<=long_end))[0]
            short_idx=np.where((time>=short_start)&(time<=short_end))[0]
            long_idx=np.setdiff1d(long_idx,short_idx)
            if len(long_idx)>=2 and len(short_idx)>0:
                diff=np.abs(np.median(wse[long_idx])-np.median(wse[short_idx]))
                if i==cur_df_id:
                    check_diff=diff
                else:
                    other_diff.append(diff)
        if check_diff is not None and len(other_diff)>0:
            if check_diff>2*np.median(other_diff) and check_diff>0.5*fluc_list[cur_df_id]:
                # too large difference, drop
                print(f"One drop in reach {cur_df_id}")
                print(f"  time: {cur_time}, wse: {cur_wse}, diff: {max_fluc}")
                print(f"  check_diff: {check_diff},other_diff: {other_diff}")
                df_reaches[cur_df_id]=df_reaches[cur_df_id].\
                    drop(index=df_reaches[cur_df_id].index[cur_idx]).\
                    reset_index(drop=True)
                # clear the check dict
                check_dict.clear()
    return df_reaches

def reach_slope_filter(df_reaches_test,slope_used='slope2'):
    df_reaches=df_reaches_test
    rplc_idx=[]
    def recalculate_slope(i):
        # recalculate the slope tobe replaced
        time=np.array(df_reaches[i]['standard_time'].tolist())
        slope=np.array(df_reaches[i][slope_used].tolist())
        valid_idx=np.sort(np.setdiff1d(np.arange(len(slope)),np.array(rplc_idx[i])))
        for j in rplc_idx[i]:
            left_idx=np.max(valid_idx[valid_idx<j],initial=-1)
            right_idx=np.min(valid_idx[valid_idx>j],initial=1e6)
            if left_idx==-1 and right_idx==1e6:
                print(f"Error: no valid index for {i} {j}")
                continue
            if left_idx==-1:
                slope[j]=slope[right_idx]
            elif right_idx==1e6:
                slope[j]=slope[left_idx]
            else:
                # Linear interpolation
                slope[j]=slope[right_idx]*(time[j]-time[left_idx])/(time[right_idx]-time[left_idx])+\
                    slope[left_idx]*(time[right_idx]-time[j])/(time[right_idx]-time[left_idx])
        df_reaches[i][slope_used]=slope
    slope_median=[]
    for df in df_reaches:
        slope=np.array(df[slope_used].tolist())
        rplc_idx.append(np.where(slope<=0)[0].tolist())
        slope_median.append(np.median(slope[slope>0]))
    print('Slope medians:',slope_median)
    for i in range(len(df_reaches)):
        recalculate_slope(i)
    check_dict={}
    while 1:
        cur_df_id=None
        cur_idx=None
        max_value=-1
        cur_time=None
        cur_slope=None
        for i in range(len(df_reaches)):
            # find the max fluctuation in the group
            time=np.array(df_reaches[i]['standard_time'].tolist())
            slope=np.array(df_reaches[i][slope_used].tolist())
            for j in range(len(time)):
                if j in rplc_idx[i] or (i,j) in check_dict:
                    continue
                cur_value=slope[j]/slope_median[i]
                if cur_value<1:
                    cur_value=1/cur_value
                if cur_value>max_value:
                    max_value=cur_value
                    cur_df_id=i
                    cur_idx=j
                    cur_time=time[j]
                    cur_slope=slope[j]
        if cur_df_id is None:
            break
        assert cur_idx<len(df_reaches[cur_df_id])
        long_start=cur_time-datetime.timedelta(days=45)
        long_end=cur_time+datetime.timedelta(days=45)
        short_start=cur_time-datetime.timedelta(days=3)
        short_end=cur_time+datetime.timedelta(days=3)
        same_start=cur_time-datetime.timedelta(hours=12)
        same_end=cur_time+datetime.timedelta(hours=12)
        check_dict[(cur_df_id,cur_idx)]=1
        other_mag=[]
        check_mag=None
        for i in range(len(df_reaches)):
            time=np.array(df_reaches[i]['standard_time'].tolist())
            slope=np.array(df_reaches[i][slope_used].tolist())
            long_idx=np.where((time>=long_start)&(time<=long_end))[0]
            if i==cur_df_id:
                short_idx=np.where((time>=same_start)&(time<=same_end))[0]
            else:
                short_idx=np.where((time>=short_start)&(time<=short_end))[0]
            long_idx=np.setdiff1d(long_idx,short_idx)
            # do we need it? we already interp
            long_idx=np.setdiff1d(long_idx,np.array(rplc_idx[i]))
            short_idx=np.setdiff1d(short_idx,np.array(rplc_idx[i]))
            if len(long_idx)>=2 and len(short_idx)>0:
                magnification=np.median(slope[short_idx])/np.median(slope[long_idx])
                if i==cur_df_id:
                    check_mag=magnification
                else:
                    other_mag.append(magnification)
        if check_mag is None:
            # no data
            continue
        replace_flag=False
        if check_mag>10 or check_mag<0.1:
            # too large difference, drop
            replace_flag=True
        elif (check_mag>3 or check_mag<1/3) and len(other_mag)>0:
            # a little bit large difference
            if check_mag<1:
                # too small is also a problem
                check_mag=1/check_mag
                other_mag=1/np.array(other_mag)
            other_median=np.median(other_mag)
            if check_mag>3*other_median:
                # not consistent with the group
                replace_flag=True
        # if cur_df_id==0 and cur_slope>0.0001:
        #     print('test ',cur_slope,check_mag,other_mag)
        if replace_flag:
            rplc_idx[cur_df_id].append(cur_idx)
            print(f"One anomoly in reach {cur_df_id}")
            print(f"  time: {cur_time}, slope: {cur_slope}")
            print(f"  check_magnification: {check_mag},other_magnification: {other_mag}")
            recalculate_slope(cur_df_id)
            check_dict.clear()
    return df_reaches
def get_neighbor_info(reach_id,up_num=2,dn_num=2):
    """
    number: number of neighbor reaches
    return:reach_id,dem,length
    """
    if up_num>0 and dn_num>0:
        rch1,dem1,len1=get_neighbor_info(reach_id,up_num=up_num,dn_num=0)
        rch2,dem2,len2=get_neighbor_info(reach_id,up_num=0,dn_num=dn_num)
        return rch1[:-1]+rch2,dem1[:-1]+dem2,len1[:-1]+len2
    dir='up' if up_num>0 else 'dn'
    maxrequest=10
    attempt_count=0
    while attempt_count < maxrequest:
        try:
            hydrocron_response = requests.get(
                f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?collection_name=SWOT_L2_HR_RiverSP_2.0&feature=Reach&feature_id={reach_id}"
                f"&start_time=2023-01-01T00:00:00Z&end_time=2025-03-20T00:00:00Z&output=csv&fields=p_lon,p_lat,reach_id,time_str,wse,width,p_length,rch_id_{dir}"
            ).json()
            if 'results' in hydrocron_response and 'csv' in hydrocron_response['results']:
                csv_str=hydrocron_response['results']['csv']
                df_reach = pd.read_csv(StringIO(csv_str))
                mask=(df_reach['time_str']!='no_data')&(df_reach['wse']>-100)&(df_reach['width']>0)
                df_reach=df_reach[mask].reset_index()
                # to be completed
                break
            else:
                attempt_count += 1
                time.sleep(1)
                #continue
        except Exception as e:
            print(f"Error occurred: {e}")
            attempt_count += 1
            time.sleep(1)
        if attempt_count==maxrequest:
            print('Exceeded maximum number of requests, unable to obtain target reach observations')
            return [],[],[]
    print('.',end='')
    pre_reach_id=None
    cur_dem=float(np.median(df_reach['wse'].tolist()))
    cur_len=float(np.median(df_reach['p_length'].tolist()))
    if up_num+dn_num==0:
        return [reach_id],[cur_dem],[cur_len]
    pre_id_list=df_reach[f'rch_id_{dir}'].tolist()
    for info in pre_id_list:
        cur_str=info+','
        first_str=cur_str.split(',')[0].strip()
        if first_str.isdigit():
            pre_reach_id=int(first_str)
            break
    if pre_reach_id is None:
        print(f"Error: {reach_id} cant find {dir} reach")
        return [reach_id],[cur_dem],[cur_len]
    rch,dem,len=get_neighbor_info(pre_reach_id,max(0,up_num-1),max(0,dn_num-1))
    if dir=='up':
        return rch+[reach_id],dem+[cur_dem],len+[cur_len]
    return [reach_id]+rch,[cur_dem]+dem,[cur_len]+len
def hydrocon_preparation(width_sub,total_neighbor_reach,total_neighbor_prior_dem,total_neighbor_length,\
        choose_index,target_reachid,raster_path_base,prior_slope=None):
    if total_neighbor_reach is None or total_neighbor_reach==[]:
        total_neighbor_reach,total_neighbor_prior_dem,total_neighbor_length=get_neighbor_info(target_reachid,up_num=2,dn_num=2)
        print(f"total_neighbor_reach: {total_neighbor_reach}")
        print(f"total_neighbor_prior_dem: {total_neighbor_prior_dem}")
        print(f"total_neighbor_length: {total_neighbor_length}")
    # 1 initialization
    neighbor_reach=[total_neighbor_reach[i] for i in choose_index]
    neighbor_prior_dem=[total_neighbor_prior_dem[i] for i in choose_index]
    neighbor_length=[total_neighbor_length[i] for i in choose_index]
    t_idx_org=int(np.where(np.array(total_neighbor_reach)==target_reachid)[0][0])
    if prior_slope is None:
        prior_slope=(total_neighbor_prior_dem[t_idx_org-1]-total_neighbor_prior_dem[t_idx_org+1])/(\
            total_neighbor_length[t_idx_org-1]/2+total_neighbor_length[t_idx_org]+total_neighbor_length[t_idx_org+1]/2)
    print("prior_slope:",prior_slope)

    # 2 get intersection dates
    number_reaches=len(neighbor_length)
    df_reaches=[]
    for i in range(number_reaches):
        print(f"pre {i} {neighbor_reach[i]}")
        df_reaches.append(get_pre_common_dates(neighbor_reach[i], neighbor_prior_dem[i],width_sub,raster_path_base, slope_used='slope2',maxrequest=10))
    date_sets = [set(df['date']) for df in df_reaches]
    pre_common_dates=set.intersection(*date_sets)
    print(f"pre_common_dates: {len(pre_common_dates)}")
    print(f"data start from: {min(pre_common_dates)}")

    # 3 get all reach data
    number_reaches=len(neighbor_length)
    df_reaches=[]
    for i in range(number_reaches):
        print(f"proc {i} {neighbor_reach[i]}")
        df_reaches.append(get_preprocessed_df(neighbor_reach[i],pre_common_dates, neighbor_prior_dem[i],width_sub,raster_path_base, slope_used='slope2',maxrequest=10))

    # 4 filters
    df_reaches=reach_slope_filter(df_reaches,slope_used='slope2')
    df_reaches=reach_group_filter(df_reaches)

    return df_reaches,prior_slope
def s3_preparation():
    pass
"""
params:
'reach_id', 'time', 'time_tai', 'time_str', 'p_lat', 'p_lon', 'river_name',
'wse', 'wse_u', 'wse_r_u', 'wse_c', 'wse_c_u',
'slope', 'slope_u', 'slope_r_u', 'slope2', 'slope2_u', 'slope2_r_u',
'width', 'width_u', 'width_c', 'width_c_u',
'area_total', 'area_tot_u', 'area_detct', 'area_det_u', 'area_wse',
'd_x_area', 'd_x_area_u',
'layovr_val', 'node_dist', 'loc_offset', 'xtrk_dist',
'dschg_c', 'dschg_c_u', 'dschg_csf', 'dschg_c_q',
'dschg_gc', 'dschg_gc_u', 'dschg_gcsf', 'dschg_gc_q',
'dschg_m', 'dschg_m_u', 'dschg_msf', 'dschg_m_q',
'dschg_gm', 'dschg_gm_u', 'dschg_gmsf', 'dschg_gm_q',
'dschg_b', 'dschg_b_u', 'dschg_bsf', 'dschg_b_q',
'dschg_gb', 'dschg_gb_u', 'dschg_gbsf', 'dschg_gb_q',
'dschg_h', 'dschg_h_u', 'dschg_hsf', 'dschg_h_q',
'dschg_gh', 'dschg_gh_u', 'dschg_ghsf', 'dschg_gh_q',
'dschg_o', 'dschg_o_u', 'dschg_osf', 'dschg_o_q',
'dschg_go', 'dschg_go_u', 'dschg_gosf', 'dschg_go_q',
'dschg_s', 'dschg_s_u', 'dschg_ssf', 'dschg_s_q',
'dschg_gs', 'dschg_gs_u', 'dschg_gssf', 'dschg_gs_q',
'dschg_i', 'dschg_i_u', 'dschg_isf', 'dschg_i_q',
'dschg_gi', 'dschg_gi_u', 'dschg_gisf', 'dschg_gi_q',
'dschg_q_b', 'dschg_gq_b',
'reach_q', 'reach_q_b',
'dark_frac', 'ice_clim_f', 'ice_dyn_f', 'partial_f', 'n_good_nod',
'obs_frac_n', 'xovr_cal_q', 'geoid_hght', 'geoid_slop',
'solid_tide', 'load_tidef', 'load_tideg', 'pole_tide',
'dry_trop_c', 'wet_trop_c', 'iono_c', 'xovr_cal_c',
'n_reach_up', 'n_reach_dn', 'rch_id_up', 'rch_id_dn',
'p_wse', 'p_wse_var', 'p_width', 'p_wid_var', 'p_n_nodes', 'p_dist_out',
'p_length', 'p_maf', 'p_dam_id', 'p_n_ch_max', 'p_n_ch_mod', 'p_low_slp',
'cycle_id', 'pass_id', 'continent_id', 'range_start_time', 'range_end_time',
'crid', 'geometry', 'sword_version', 'collection_shortname', 'collection_version',
'granuleUR', 'ingest_time'
"""