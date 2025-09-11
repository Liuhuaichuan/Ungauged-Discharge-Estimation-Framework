#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from MetroManVariables import Recalculation
import datetime
from calcnhat import calcnhat
from numpy import sqrt

def excel_num_to_date(excel_num):
    excel_epoch = datetime.datetime(1899, 12, 30)
    delta = datetime.timedelta(days=excel_num)
    return excel_epoch + delta
    
#%%
def Recalculate_product(Allobs,SoS_parms, E_hy, product_first_day,target_reach_index,nopt=5):

    if (SoS_parms.sosA0[target_reach_index]<0) | (np.isnan(SoS_parms.sosA0[target_reach_index])):
        print('No A0 for target reach')
        return None
    # read domain
    Rel=Recalculation()

    index_before= E_hy.date_t<=product_first_day
    index_after= E_hy.date_t>product_first_day

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
            dA1_before_obs[i]=(width_before[i]*2)*(wse_before[i]-wse_before[i])/2
            dA1_before_fit[i]=(fitted_width_before[i]*2)*(wse_before[i]-wse_before[i])/2
        else:
            dA1_before_obs[i]=(width_before[i]+width_before[i+1])*(wse_before[i]-wse_before[i+1])/2
            dA1_before_fit[i]=(fitted_width_before[i]+fitted_width_before[i+1])*(wse_before[i]-wse_before[i+1])/2

    for i in range(len(width_before)-1,-1,-1):
        dA2_before_obs[i]=np.sum(dA1_before_obs[i:])
        dA2_before_fit[i]=np.sum(dA1_before_fit[i:])

    #print('dA1_before',dA1_before)
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
            if len(width_before)==0:
                dA1_after_obs[i]=0
                dA1_after_fit[i]=0
            else:
                dA1_after_obs[i]=(width_after[i]+width_before[-1])*(wse_after[i]-wse_before[-1])/2
                dA1_after_fit[i]=(fitted_width_after[i]+fitted_width_before[-1])*(wse_after[i]-wse_before[-1])/2
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
    print('dA2_after_fit',dA2_after_fit)

    Rel.date_t=E_hy.date_t
    Rel.dA_obs=dA2_obs
    Rel.dA_fit=dA2_fit
    
    nhat_obs=calcnhat(E_hy.obs_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               SoS_parms.sosA0[target_reach_index]+Rel.dA_obs, \
                               SoS_parms.sosx1[target_reach_index],SoS_parms.sosna[target_reach_index],nopt)
    nhat_fit=calcnhat(E_hy.fitted_w, E_hy.h, Allobs.hmin[target_reach_index], \
                               SoS_parms.sosA0[target_reach_index]+Rel.dA_fit, \
                               SoS_parms.sosx1[target_reach_index],SoS_parms.sosna[target_reach_index],nopt)
    print('nhat_fit',nhat_fit)
    print('A_fit',SoS_parms.sosA0[target_reach_index]+Rel.dA_fit)
    #print('nahat',nhat)
    Rel.Q_obs=1/nhat_obs*(SoS_parms.sosA0[target_reach_index]+Rel.dA_obs)**(5/3) * E_hy.obs_w**(-2/3) * E_hy.S**0.5
    Rel.Q_fit=1/nhat_fit*(SoS_parms.sosA0[target_reach_index]+Rel.dA_fit)**(5/3) * E_hy.fitted_w**(-2/3) * E_hy.S**0.5
    Rel.Q_obs[SoS_parms.sosA0[target_reach_index]+Rel.dA_obs<0]=np.nan
    Rel.Q_fit[SoS_parms.sosA0[target_reach_index]+Rel.dA_fit<0]=np.nan

    return Rel