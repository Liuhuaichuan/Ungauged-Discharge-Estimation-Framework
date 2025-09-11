import numpy as np
# import xarray as xr
import netCDF4 as nc
from MetroManVariables import SoS_calibration
import datetime

def Getsos_calibration(DAll,all_reachid, target_reachid, prior_path=r'D:\Geography dataset\SWORD\SoS\Update_202404\na_sword_v16_SOS_unconstrained_0001_20240611T010141_priors.nc'):
    SoS_param=SoS_calibration(DAll)
    # Input the paths of priors and results (discharge)
    # NA for test,based on SWORDv16
    result_path=prior_path.replace('priors','results')
    prior_nc=nc.Dataset(prior_path,format='NETCDF4')
    result_nc=nc.Dataset(result_path,format='NETCDF4')
    
    Dis_Algorithm=['hivdi','neobam','momma','metroman','sad','sic4dvar']
    Gauge_Agency='USGS'
    sos_reachids_rs=result_nc['reaches']['reach_id'][:]
    sos_reachids_pr=prior_nc['reaches']['reach_id'][:]
    sos_lons_rs=result_nc['reaches']['x'][:]
    sos_lats_rs=result_nc['reaches']['y'][:]
    
    sos_A0hats_rs=result_nc['metroman']['A0hat'][:]
    sos_nahats_rs=result_nc['metroman']['nahat'][:]
    sos_x1hats_rs=result_nc['metroman']['x1hat'][:]
    
    sos_metroqs_rs=result_nc['metroman']['allq'][:]
    sos_mommaqs_rs=result_nc['momma']['Q'][:]
    sos_times_rs=result_nc['reaches']['time'][:]
    
    sos_momma_B_rs=result_nc['moi']['momma']['B'][:]
    sos_momma_H_rs=result_nc['moi']['momma']['H'][:]
    sos_momma_n_rs=result_nc['momma']['n'][:]

    for i, reachid in enumerate(all_reachid):
        river_indexes=np.where(sos_reachids_rs==int(reachid))[0]
        A0=sos_A0hats_rs.filled()[river_indexes[0]]
        na=sos_nahats_rs.filled()[river_indexes[0]]
        x1=sos_x1hats_rs.filled()[river_indexes[0]]
        SoS_param.sosA0[i,0]=A0
        SoS_param.sosna[i,0]=na
        SoS_param.sosx1[i,0]=x1
    
    SWOT_gauge_reaches=prior_nc[Gauge_Agency][f'{Gauge_Agency}_reach_id'][:].filled()
    gauge_index=np.where(SWOT_gauge_reaches==int(target_reachid))[0]#必须要保证该reach有实测水位
    try:
        gauge_time=prior_nc[Gauge_Agency][f'{Gauge_Agency}_qt'][gauge_index].filled().astype(int)[0]
        gauge_discharge=prior_nc[Gauge_Agency][f'{Gauge_Agency}_q'][gauge_index].filled()[0]
    except:
        gauge_time= np.array([])
        gauge_discharge= np.array([])
    nonmissing_index= (gauge_discharge>=0) & (gauge_time>1000) & (gauge_discharge<100000)
    gauge_discharge=gauge_discharge[nonmissing_index]
    gauge_time=gauge_time[nonmissing_index]
    gauge_time_datetime=np.array([datetime.datetime.fromordinal(gt) for gt in gauge_time])

    river_indexes=np.where(sos_reachids_rs==int(target_reachid))[0]
    SoS_param.WBM_discharge=prior_nc['model']['mean_q'][:].filled()[river_indexes[0]]
    SoS_param.gaugetime=gauge_time_datetime
    SoS_param.gaugedisg=gauge_discharge
    
    #获取产品流量
    selected_discharge=sos_metroqs_rs[river_indexes[0]]
    selected_time=sos_times_rs[river_indexes[0]]
    if len(selected_discharge)==len(selected_time):
        nonnan_index= (selected_discharge>=0) & (selected_time>=0) & (selected_discharge<500000)
        selected_discharge=selected_discharge[nonnan_index]
        selected_time=selected_time[nonnan_index]
    else:
        selected_time=np.array([])
        selected_discharge=np.array([])
    time_datetime=[]
    swot_ts=datetime.datetime(2000,1,1,0,0,0)
    for t in selected_time:
        #t_str=(swot_ts+datetime.timedelta(seconds=t)).strftime('%Y-%m-%dT%H:%M:%S')
        time_datetime.append(swot_ts+datetime.timedelta(seconds=t))
    #time_str_np=np.array(time_str)
    time_datetime=np.array(time_datetime)
    SoS_param.product_time=time_datetime
    SoS_param.product_disg=selected_discharge
    if len(sos_times_rs[river_indexes[0]])>0:
        t0=sos_times_rs[river_indexes[0]][0]
        if t0<0:
            t0=0
        SoS_param.initial_time=swot_ts+datetime.timedelta(seconds=t0)
    else:
        SoS_param.initial_time=np.nan
    return SoS_param