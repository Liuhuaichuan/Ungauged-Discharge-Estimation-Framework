#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array,diff,ones,reshape
from MetroManVariables import Domain,Observations

#%%
def ReadObs(fname,begin_limit=1):

    fid=open(fname,"r")
    infile=fid.readlines()
    
    # read domain
    D=Domain()
    D.nR=eval(infile[1])
    buf=infile[3]; buf=buf.split(); D.xkm=array(buf,float)
    buf=infile[5]; buf=buf.split(); D.L=array(buf,float)
    D.nt=eval(infile[7])
    buf=infile[9]; buf=buf.split(); D.t=array([buf],float)

    # filter, must after begin limit
    begin_mask=(D.t>=begin_limit)
    D.t=reshape(D.t[begin_mask],(1,-1,))
    begin_mask=reshape(begin_mask,(-1,))
    D.nt=sum(begin_mask)
    print('nt=',D.nt)

    
    #%% read observations   
    Obs=Observations(D)
    for i in range(0,D.nR):
        buf=infile[i+11]
        buf=buf.split()
        Obs.h[i,:]=array(buf,float)[begin_mask]
    buf=infile[12+D.nR]
    buf=buf.split()
    Obs.h0=array([buf],float)
    for i in range(0,D.nR):
        buf=infile[14+D.nR+i]
        buf=buf.split()
        Obs.S[i,:]=array(buf,float)[begin_mask]
    for i in range(0,D.nR):
        buf=infile[15+D.nR*2+i]
        buf=buf.split()
        Obs.w[i,:]=array(buf,float)[begin_mask]
    Obs.sigS=eval(infile[16+D.nR*3])
    Obs.sigh=eval(infile[18+D.nR*3])
    Obs.sigw=eval(infile[20+D.nR*3])
    
    #D.dt=reshape((diff(D.t).T*86400 * ones((1,D.nR))).T,(D.nR*(D.nt-1),1))
    D.dt = reshape((diff(D.t).T * 86400 * ones((1, D.nR))).T, (D.nR * (D.nt-1), 1))
    #%% create resahepd versions of observations
    Obs.hv=reshape(Obs.h, (D.nR*D.nt,1) )
    Obs.Sv=reshape(Obs.S, (D.nR*D.nt,1) )
    Obs.wv=reshape(Obs.w, (D.nR*D.nt,1) )
    
    #%%
    fid.close()   

    return D,Obs