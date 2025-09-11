#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from MetroManVariables import Estimates_hydrocorn
import datetime
from calcnhat import calcnhat
from numpy import sqrt
from find_peak_position import find_peak_position

def excel_num_to_date(excel_num):
    excel_epoch = datetime.datetime(1899, 12, 30)
    delta = datetime.timedelta(days=excel_num)
    return excel_epoch + delta
    
#%%
def CalculateEstimates_hydrocorn(df_reach,Allobs,Chain,Estimate, Prior,SWOT_first_day,target_reach_index,nopt,use_peak=False,print_flag=False):

    # read domain
    E_hy=Estimates_hydrocorn()
    E_hy.nt=len(df_reach)
    E_hy.date_t = df_reach['date']
    E_hy.h=np.array(df_reach['wse'])
    E_hy.fitted_w=np.array(df_reach['fitted_width'])
    E_hy.obs_w=np.array(df_reach['width'])
    E_hy.S=np.array(df_reach['slope2'])

    index_before= E_hy.date_t<=SWOT_first_day
    index_after= E_hy.date_t>SWOT_first_day

    date_before= E_hy.date_t[index_before]
    wse_before= E_hy.h[index_before]
    width_before= E_hy.obs_w[index_before]
    fitted_width_before= E_hy.fitted_w[index_before]
    dA1_before_obs=np.zeros(len(width_before))
    dA1_before_fit=np.zeros(len(width_before))
    dA2_before_obs=np.zeros(len(width_before))
    dA2_before_fit=np.zeros(len(width_before))
    
    for i in range(len(width_before)-1,-1,-1):
        if i==len(width_before)-1:
            dA1_before_obs[i]=(width_before[i]+Allobs.w[target_reach_index,0])*(wse_before[i]-Allobs.h[target_reach_index,0])/2
            dA1_before_fit[i]=(fitted_width_before[i]+Allobs.w[target_reach_index,0])*(wse_before[i]-Allobs.h[target_reach_index,0])/2
        else:
            dA1_before_obs[i]=(width_before[i]+width_before[i+1])*(wse_before[i]-wse_before[i+1])/2
            dA1_before_fit[i]=(fitted_width_before[i]+fitted_width_before[i+1])*(wse_before[i]-wse_before[i+1])/2

    for i in range(len(width_before)-1,-1,-1):
        dA2_before_obs[i]=np.sum(dA1_before_obs[i:])
        dA2_before_fit[i]=np.sum(dA1_before_fit[i:])

    #print('dA1_before',dA1_before)
    if print_flag:
        print('dA2_before_fit',dA2_before_fit)
    date_after= E_hy.date_t[index_after]
    wse_after= E_hy.h[index_after]
    width_after= E_hy.obs_w[index_after]
    fitted_width_after= E_hy.fitted_w[index_after]
    dA1_after_obs=np.zeros(len(width_after))
    dA1_after_fit=np.zeros(len(width_after))
    dA2_after_obs=np.zeros(len(width_after))
    dA2_after_fit=np.zeros(len(width_after))

    for i in range(0,len(width_after)):
        if i==0:
            dA1_after_obs[i]=(width_after[i]+Allobs.w[target_reach_index,0])*(wse_after[i]-Allobs.h[target_reach_index,0])/2
            dA1_after_fit[i]=(fitted_width_after[i]+Allobs.w[target_reach_index,0])*(wse_after[i]-Allobs.h[target_reach_index,0])/2
        else:
            dA1_after_obs[i]=(width_after[i]+width_after[i-1])*(wse_after[i]-wse_after[i-1])/2
            dA1_after_fit[i]=(fitted_width_after[i]+fitted_width_after[i-1])*(wse_after[i]-wse_after[i-1])/2

    for i in range(0,len(width_after)):
        dA2_after_obs[i]=np.sum(dA1_after_obs[:i+1])
        dA2_after_fit[i]=np.sum(dA1_after_fit[:i+1])

    dA1_obs=np.concatenate((dA1_before_obs, dA1_after_obs))
    dA2_obs=np.concatenate((dA2_before_obs, dA2_after_obs))

    dA1_fit=np.concatenate((dA1_before_fit, dA1_after_fit))
    dA2_fit=np.concatenate((dA2_before_fit, dA2_after_fit))
    #print('dA1_after',dA1_after)
    if print_flag:
        print('dA2_after_fit',dA2_after_fit)
    
    E_hy.dA_obs=dA2_obs
    E_hy.dA_fit=dA2_fit
    # Estimate Manning coefficient
    nhat_obs=calcnhat(E_hy.obs_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               Estimate.A0hat[target_reach_index]+E_hy.dA_obs, \
                               Estimate.x1hat[target_reach_index],Estimate.nahat[target_reach_index],nopt)
    nhat_fit=calcnhat(E_hy.fitted_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               Estimate.A0hat[target_reach_index]+E_hy.dA_fit, \
                               Estimate.x1hat[target_reach_index],Estimate.nahat[target_reach_index],nopt)
    if print_flag:
        print('nhat_fit',nhat_fit)
        print('A_fit',Estimate.A0hat[target_reach_index]+E_hy.dA_fit)
    #print('nahat',nhat)
    E_hy.Q_obs=1/nhat_obs*(Estimate.A0hat[target_reach_index]+E_hy.dA_obs)**(5/3) * E_hy.obs_w**(-2/3) * E_hy.S**0.5
    E_hy.Q_fit=1/nhat_fit*(Estimate.A0hat[target_reach_index]+E_hy.dA_fit)**(5/3) * E_hy.fitted_w**(-2/3) * E_hy.S**0.5
    E_hy.Q_obs[Estimate.A0hat[target_reach_index]+E_hy.dA_obs<0]=np.nan
    E_hy.Q_fit[Estimate.A0hat[target_reach_index]+E_hy.dA_fit<0]=np.nan

    nhat_all_obs=np.zeros((Chain.N,E_hy.nt))
    nhat_all_fit=np.zeros((Chain.N,E_hy.nt))
    E_hy.thetaQ_obs=np.zeros((Chain.N,E_hy.nt))
    E_hy.thetaQ_fit=np.zeros((Chain.N,E_hy.nt))
    for i in range(0,Chain.N):
        nhat_all_fit[i,:]=calcnhat(E_hy.fitted_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               Estimate.A0hat[target_reach_index]+E_hy.dA_fit, \
                               Chain.thetax1[target_reach_index,i],Chain.thetana[target_reach_index,i],nopt)
        nhat_all_obs[i,:]=calcnhat(E_hy.obs_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               Estimate.A0hat[target_reach_index]+E_hy.dA_obs, \
                               Chain.thetax1[target_reach_index,i],Chain.thetana[target_reach_index,i],nopt)
        
        E_hy.thetaQ_obs[i,:]=1/nhat_all_obs[i,:]*(Chain.thetaA0[target_reach_index,i]\
                                  +E_hy.dA_obs)**(5/3) * E_hy.obs_w**(-2/3) * E_hy.S**0.5
        E_hy.thetaQ_fit[i,:]=1/nhat_all_fit[i,:]*(Chain.thetaA0[target_reach_index,i]\
                                  +E_hy.dA_fit)**(5/3) * E_hy.fitted_w**(-2/3) * E_hy.S**0.5
        E_hy.thetaQ_obs[i,Chain.thetaA0[target_reach_index,i]+E_hy.dA_obs<0]=np.nan
        E_hy.thetaQ_fit[i,Chain.thetaA0[target_reach_index,i]+E_hy.dA_fit<0]=np.nan

    E_hy.stdQ_obs=np.nanstd(E_hy.thetaQ_obs[Chain.Nburn:,:],0)
    E_hy.stdQ_fit=np.nanstd(E_hy.thetaQ_fit[Chain.Nburn:,:],0)
    E_hy.meanQ_obs=np.nanmean(E_hy.thetaQ_obs[Chain.Nburn:,:],0)
    E_hy.meanQ_fit=np.nanmean(E_hy.thetaQ_fit[Chain.Nburn:,:],0)
    if use_peak:
        E_hy.Qpeak_obs=np.array([find_peak_position(E_hy.thetaQ_obs[Chain.Nburn:,i]) for i in range(E_hy.nt)])

    nhat_obs=calcnhat(E_hy.obs_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               Prior.meanA0[target_reach_index]+E_hy.dA_obs, \
                               Prior.meanx1[target_reach_index],Prior.meanna[target_reach_index],nopt)
    nhat_fit=calcnhat(E_hy.fitted_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               Prior.meanA0[target_reach_index]+E_hy.dA_fit, \
                               Prior.meanx1[target_reach_index],Prior.meanna[target_reach_index],nopt)
    E_hy.Q_pobs=1/nhat_obs * (Prior.meanA0[target_reach_index]+E_hy.dA_obs)**(5/3) \
                *E_hy.obs_w**(-2/3)*sqrt(E_hy.S);
    E_hy.Q_pfit=1/nhat_fit * (Prior.meanA0[target_reach_index]+E_hy.dA_fit)**(5/3) \
                *E_hy.fitted_w**(-2/3)*sqrt(E_hy.S);

    E_hy.Q_pobs[Prior.meanA0[target_reach_index]+E_hy.dA_obs<0]=np.nan
    E_hy.Q_pfit[Prior.meanA0[target_reach_index]+E_hy.dA_fit<0]=np.nan
    return E_hy