#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from MetroManVariables import S3_reconstruction
import datetime
from calcnhat import calcnhat

def excel_num_to_date(excel_num):
    excel_epoch = datetime.datetime(1899, 12, 30)
    delta = datetime.timedelta(days=excel_num)
    return excel_epoch + delta
    
#%%
def ReadS3(fname, DAll, Allobs, Estimate, Chain, SWOT_first_day, target_reach_index, nopt):

    fid=open(fname,"r")
    infile=fid.readlines()
    
    # read domain
    S3_constru=S3_reconstruction()
    S3_constru.nodeid=eval(infile[1])
    S3_constru.nt=eval(infile[3])
    S3_constru.measured_slope=eval(infile[5])
    S3_constru.prior_slope=eval(infile[7])
    
    buf=infile[9]; buf=buf.split(); S3_constru.excel_t=np.array(buf, float)
    #print(S3_constru.excel_t)
    S3_constru.date_t=np.array([excel_num_to_date(excel_t) for excel_t in S3_constru.excel_t])
    
    buf=infile[11]; buf=buf.split(); S3_constru.ori_h=np.array(buf,float)
    buf=infile[13]; buf=buf.split(); S3_constru.cor_h=np.array(buf,float)
    buf=infile[15]; buf=buf.split(); S3_constru.fitted_w=np.array(buf,float)

    SWOT_datetime = np.array([SWOT_first_day + datetime.timedelta(days=t-1) for t in DAll.t[0]])
    index_before= S3_constru.date_t<=SWOT_first_day
    index_after= S3_constru.date_t>SWOT_first_day

    date_before= S3_constru.date_t[index_before]
    wse_before= S3_constru.cor_h[index_before]
    width_before= S3_constru.fitted_w[index_before]
    dA1_before=np.zeros(len(width_before))
    dA2_before=np.zeros(len(width_before))
    for i in range(len(width_before)-1,-1,-1):
        if i==len(width_before)-1:
            dA1_before[i]=(width_before[i]+Allobs.w[target_reach_index,0])*(wse_before[i]-Allobs.h[target_reach_index,0])/2
        else:
            dA1_before[i]=(width_before[i]+width_before[i+1])*(wse_before[i]-wse_before[i+1])/2

    for i in range(len(width_before)-1,-1,-1):
        dA2_before[i]=np.sum(dA1_before[i:])

    #print('dA1_before',dA1_before)
    print('dA2_before',dA2_before)
    date_after= S3_constru.date_t[index_after]
    wse_after= S3_constru.cor_h[index_after]
    width_after= S3_constru.fitted_w[index_after]
    dA1_after=np.zeros(len(width_after))
    dA2_after=np.zeros(len(width_after))
    for i in range(0,len(width_after)):
        if i==0:
            dA1_after[i]=(width_after[i]+Allobs.w[target_reach_index,0])*(wse_after[i]-Allobs.h[target_reach_index,0])/2
        else:
            dA1_after[i]=(width_after[i]+width_after[i-1])*(wse_after[i]-wse_after[i-1])/2

    for i in range(0,len(width_after)):
        dA2_after[i]=np.sum(dA1_after[:i+1])

    dA1=np.concatenate((dA1_before, dA1_after))
    dA2=np.concatenate((dA2_before, dA2_after))
    #print('dA1_after',dA1_after)
    print('dA2_after',dA2_after)
    
    S3_constru.dA=dA2
    nhat=calcnhat(S3_constru.fitted_w, S3_constru.cor_h, Allobs.hmin[target_reach_index], \
                               Estimate.A0hat[target_reach_index]+S3_constru.dA, \
                               Estimate.x1hat[target_reach_index],Estimate.nahat[target_reach_index],nopt)
    print('nhat',nhat)
    print('A',Estimate.A0hat[target_reach_index]+S3_constru.dA)
    #print('nahat',nhat)
    S3_constru.Q_ms=1/nhat*(Estimate.A0hat[target_reach_index]+S3_constru.dA)**(5/3) * S3_constru.fitted_w**(-2/3) * S3_constru.measured_slope**0.5
    S3_constru.Q_ps=1/nhat*(Estimate.A0hat[target_reach_index]+S3_constru.dA)**(5/3) * S3_constru.fitted_w**(-2/3) * S3_constru.prior_slope**0.5
    S3_constru.Q_ms[Estimate.A0hat[target_reach_index]+S3_constru.dA<0]=np.nan
    S3_constru.Q_ps[Estimate.A0hat[target_reach_index]+S3_constru.dA<0]=np.nan

    nhat_all=np.zeros((Chain.N,S3_constru.nt))
    S3_constru.thetaQ_ms=np.zeros((Chain.N,S3_constru.nt))
    S3_constru.thetaQ_ps=np.zeros((Chain.N,S3_constru.nt))
    for i in range(0,Chain.N):
        nhat_all[i,:]=calcnhat(S3_constru.fitted_w, S3_constru.cor_h, Allobs.hmin[target_reach_index], \
                               Estimate.A0hat[target_reach_index]+S3_constru.dA, \
                               Chain.thetax1[target_reach_index,i],Chain.thetana[target_reach_index,i],nopt)
        S3_constru.thetaQ_ms[i,:]=1/nhat_all[i,:]*(Chain.thetaA0[target_reach_index,i]\
                                  +S3_constru.dA)**(5/3) * S3_constru.fitted_w**(-2/3) * S3_constru.measured_slope**0.5
        S3_constru.thetaQ_ps[i,:]=1/nhat_all[i,:]*(Chain.thetaA0[target_reach_index,i]\
                                  +S3_constru.dA)**(5/3) * S3_constru.fitted_w**(-2/3) * S3_constru.prior_slope**0.5
        S3_constru.thetaQ_ms[i,Chain.thetaA0[target_reach_index,i]+S3_constru.dA<0]=np.nan
        S3_constru.thetaQ_ps[i,Chain.thetaA0[target_reach_index,i]+S3_constru.dA<0]=np.nan

    S3_constru.stdQ_ms=np.nanstd(S3_constru.thetaQ_ms[Chain.Nburn:,:],0)
    S3_constru.stdQ_ps=np.nanstd(S3_constru.thetaQ_ps[Chain.Nburn:,:],0)
    S3_constru.meanQ_ms=np.nanmean(S3_constru.thetaQ_ms[Chain.Nburn:,:],0)
    S3_constru.meanQ_ps=np.nanmean(S3_constru.thetaQ_ps[Chain.Nburn:,:],0)

    fid.close()   

    return S3_constru