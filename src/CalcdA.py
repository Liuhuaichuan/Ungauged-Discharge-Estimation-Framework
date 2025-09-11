#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import empty,reshape,triu,ones,zeros,concatenate
from CalcU import CalcU

def CalcdA(D,Obs):
    
    DeltaAHat=empty( (D.nR,D.nt-1) )
    
    for r in range(0,D.nR):
        for t in range(0,D.nt-1):
            DeltaAHat[r,t]=(Obs.w[r,t]+Obs.w[r,t+1])/2 * (Obs.h[r,t+1]-Obs.h[r,t])
     
    # changed how this part works compared with Matlab, avoiding translating calcU
    DeltaAHatv=reshape(DeltaAHat,(D.nR*(D.nt-1),1) )
    
    Obs.dA= concatenate(  (zeros( (D.nR,1) ), DeltaAHat @ triu(ones( (D.nt-1,D.nt-1) ),0)),1 ) #DeltaAHat @ triu(ones( (D.nt-1,D.nt-1) ),0))，维度(D.nR,D.nt-1),但为DA的累积量
            #concatenate(..., 1) -
    U=CalcU(D)
    Obs.dAv=U @ DeltaAHatv

    # Note that the below commented command is equivalent, but "U" is needed in
    #   other functions, and this is a good test for consistency
    #Obs.dAv=reshape(Obs.dA, (D.nR*D.nt,1)) #(D.nR*D.nt,1)，每个reach的累计DA，第一个值为0
    
    return Obs
    