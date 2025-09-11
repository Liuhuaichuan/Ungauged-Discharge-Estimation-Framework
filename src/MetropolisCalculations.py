#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:57:11 2020

@author: mtd
"""

from numpy import empty,mean,exp,putmask,log,any
from scipy.stats import lognorm
import numpy as np
import time
from CalcDelta import CalcDelta
from CalcADelta import CalcADelta
from CalcB import CalcB
from logninvstat import logninvstat
from CalcLklhd import CalcLklhd
from numpy.random import seed,rand,randn

def MetropolisCalculations(Prior,D,Obs,jmp,C,R,DAll,AllObs,nOpt,DebugMode,A0_max=6000,output_path=None):
    [Delta,DeltaA,B,C,thetauA0,thetauna,thetaux1,thetauq,R]=InitializeMetropolis(D,C,Prior,R)
    output_buffer = []
    if DebugMode:
        C.N=int(C.N/10)
        C.Nburn=int(C.Nburn/10)
    jmp.stdA0=0.1*mean(thetauA0)
    jmp.stdna=0.03*mean(thetauna)
    jmp.stdx1=0.1*mean(thetaux1)
    
    # set target acceptance rates to 0.25 since all quantities are vectors (length D.nR)
    jmp.target1=0.25
    jmp.target2=0.25
    jmp.target3=0.25
    
    jmp.stdA0s=empty((C.N))
    if output_path is not None:
        output_buffer.append(f"initial std A0: {jmp.stdA0}")
    jmp.stdnas=empty((C.N))
    if output_path is not None:
        output_buffer.append(f"initial std na: {jmp.stdna}")
    jmp.stdx1s=empty((C.N))
    if output_path is not None:
        output_buffer.append(f"initial std x1: {jmp.stdx1}")
    
    meanA0=Prior.meanA0
    covA0=Prior.stdA0/meanA0
    vA0=(covA0*meanA0)**2
    [muA0,sigmaA0]=logninvstat(meanA0,vA0)
    
    #%%
    meanna=Prior.meanna
    covna=Prior.stdna/meanna
    vna=(covna*Prior.meanna)**2
    [muna,sigmana] = logninvstat(meanna,vna)
    
    meanx1=Prior.meanx1# n.opt<5
    covx1=Prior.stdx1/meanx1
    vx1=(covx1*Prior.meanx1)**2
    [mux1,sigmax1] = logninvstat(np.abs(-Prior.meanx1),vx1)

    pu1=np.ones(D.nR)#lognorm.pdf(thetauA0,sigmaA0,0,exp(muA0))
    if output_path is not None:
        output_buffer.append(f"pu1: {pu1}")
    pu2=lognorm.pdf(thetauna,sigmana,0,exp(muna))
    if output_path is not None:
        output_buffer.append(f"pu2: {pu2}")
    
    if nOpt<5:
        pu3=lognorm.pdf(-thetaux1,sigmax1,0,exp(mux1))
    elif nOpt==5:
        pu3=lognorm.pdf(thetaux1,sigmax1,0,exp(mux1))
    if output_path is not None:
        output_buffer.append(f"pu3: {pu3}")
    fu, Theta,Cf_last,eps_last=CalcLklhd(Obs,AllObs,thetauA0,thetauna,thetaux1,D,Prior,Delta,DeltaA,B,thetauq,nOpt)# logLike
    if output_path is not None:
        output_buffer.append(f"fu: {fu}")
    C.n_a1=0
    C.n_a2=0
    C.n_a3=0
    
    C.Like=empty((C.N))
    C.LogLike=empty((C.N))
    C.Theta=empty((D.nR*(D.nt-1),C.N))

    C.Cf_mean=empty((C.N))
    C.Cf_cond=empty((C.N))
    C.Cf_det=empty((C.N))
    
    eps_record=empty((C.Nburn))
    
    #%%
    tic=time.process_time()
    A0_pointer=0
    def move_pointer(A0_pointer):
        A0_pointer+=1
        if A0_pointer>=C.N*5:
            A0_pointer=0
            R.z1=randn(D.nR,C.N*5)
        return A0_pointer
    static_eps=None
    for i in range(0,C.N):
        if i==C.Nburn:
            static_eps=mean(eps_record)
            print(f"static eps value: {static_eps}")
        if output_path is not None:
            output_buffer.append(f"Iteration # {i+1}/{C.N}.")
        if i%1000==0:
            print(f"Iteration # {i+1}/{C.N}, A0 pointer at {A0_pointer}")
        if i<C.Nburn and i>0 and i%100==0:
            jmp.stdA0=mean(jmp.stdA0s[0:i-1] )/jmp.target1*(C.n_a1/i)
            jmp.stdna=mean(jmp.stdnas[0:i-1] )/jmp.target2*(C.n_a2/i)
            jmp.stdx1=mean(jmp.stdx1s[0:i-1] )/jmp.target3*(C.n_a3/i)

        jmp.stdA0s[i]=jmp.stdA0    
        jmp.stdnas[i]=jmp.stdna
        jmp.stdx1s[i]=jmp.stdx1
        
        #A0
        
        # uniform distribution
        cur_A0_min=jmp.A0min.reshape((D.nR,))
        cur_A0_max=A0_max.reshape((D.nR,))
        # uniform_A0=uniform_A0_min+(uniform_A0_max-uniform_A0_min)*R.z1_uni[:,i]

        # current_rand_min=(jmp.A0min.reshape((D.nR,))-thetauA0)/jmp.stdA0
        # current_rand_max=(A0_max-thetauA0)/jmp.stdA0

        # normal distribution
        thetavA0=thetauA0+jmp.stdA0*R.z1[:,A0_pointer]
        A0_pointer=move_pointer(A0_pointer)

        while 1:
            range_mask= (thetavA0 < cur_A0_min) | (thetavA0 > cur_A0_max)
            if not any(range_mask):
                break
            new_A0=thetauA0+jmp.stdA0*R.z1[:,A0_pointer]
            A0_pointer=move_pointer(A0_pointer)
            thetavA0[range_mask]=new_A0[range_mask]

        # thetavA0[thetavA0<jmp.A0min.reshape((D.nR,))]=putmask(thetavA0,thetavA0<jmp.A0min,jmp.A0min)#不要小于A0的最小值
        # thetavA0[thetavA0>A0_max]=putmask(thetavA0,thetavA0>A0_max,A0_max)
        #thetavA0[thetavA0>jmp.A0min.reshape((D.nR,))*2]=putmask(thetavA0,thetavA0>jmp.A0min*2,jmp.A0min*2)
        
        #pv1=lognorm.pdf(thetavA0,sigmaA0,0,exp(muA0))
        pv1=np.ones(D.nR)
        fv, Theta,_,__=CalcLklhd(Obs,AllObs,thetavA0,thetauna,thetaux1,D,Prior,Delta,DeltaA,B,thetauq,nOpt,static_eps)
        
        MetRatio=exp(fv-fu)*exp(sum(log(pv1))-sum(log(pu1)))
        '''
        if exp(fv-fu)>=1:
            MetRatio=0.8
        else:
            MetRatio=0.2
            '''
        if output_path is not None:
            output_buffer.append(f"A0 {MetRatio}")
        if MetRatio > R.u1[i]:
            C.n_a1=C.n_a1+1
            thetauA0=thetavA0
            fu=fv
            pu1=pv1 # update u->v
            Cf_last=_
            eps_last=__
        C.thetaA0[:,i]=thetauA0.T
        
        #na
        thetavna=thetauna+jmp.stdna*R.z2[:,i]
        thetavna[thetavna<jmp.nmin]=putmask(thetavna,thetavna<jmp.nmin,jmp.nmin)
        thetavna[thetavna>1]=putmask(thetavna,thetavna>1,0.99)
        
        pv2=lognorm.pdf(thetavna,sigmana,0,exp(muna))
        #pv2=np.ones(D.nR)
        fv, Theta,_,__=CalcLklhd(Obs,AllObs,thetauA0,thetavna,thetaux1,D,Prior,Delta,DeltaA,B,thetauq,nOpt,static_eps)
        
        MetRatio=exp(fv-fu)*exp(sum(log(pv2))-sum(log(pu2)))
        '''
        if exp(fv-fu)*exp(sum(log(pv2))-sum(log(pu2)))>=1:
            MetRatio=0.8
        else:
            MetRatio=0.2
            '''
        if output_path is not None:
            output_buffer.append(f"na {MetRatio}")
        if MetRatio > R.u2[i]:
            C.n_a2=C.n_a2+1
            thetauna=thetavna
            fu=fv
            pu2=pv2
            Cf_last=_
            eps_last=__
        C.thetana[:,i]=thetauna.T
        
        #x1
        thetavx1=thetaux1+jmp.stdx1*R.z3[:,i]
        
        if nOpt<5:
            thetavx1[thetavx1>=0]=putmask(thetavx1,thetavx1>=0,-0.1)
            pv3=lognorm.pdf(-thetavx1,sigmax1,0,exp(mux1))
            #pv3=np.ones(D.nR)
        elif nOpt==5:
            thetavx1[thetavx1<=0]=putmask(thetavx1,thetavx1<=0,0.1)
            #thetavx1[thetavx1>1]=putmask(thetavx1,thetavx1>1,1)
            pv3=lognorm.pdf(thetavx1,sigmax1,0,exp(mux1))
            #pv3=np.ones(D.nR)

        fv, Theta,_,__=CalcLklhd(Obs,AllObs,thetauA0,thetauna,thetavx1,D,Prior,Delta,DeltaA,B,thetauq,nOpt,static_eps)
        
        if any(pv3==0):
            MetRatio=0
        else:
            MetRatio=exp(fv-fu)*exp(sum(log(pv3))-sum(log(pu3)))
            '''
            if exp(fv-fu)*exp(sum(log(pv3))-sum(log(pu3)))>=1:
                MetRatio=0.8
            else:
                MetRatio=0.2
                '''
        if output_path is not None:
            output_buffer.append(f"x1 {MetRatio}")
        if MetRatio > R.u3[i]:
            C.n_a3=C.n_a3+1
            thetaux1=thetavx1
            fu=fv
            pu3=pv3
            Cf_last=_
            eps_last=__
        C.thetax1[:,i]=thetaux1.T
        if output_path is not None:
            output_buffer.append(f"fu: {fu}")
        C.Like[i]=exp(fu)
        C.LogLike[i]=fu
        C.Theta[:,i]=Theta.reshape((D.nR*(D.nt-1),))


        C.Cf_mean[i]=mean(Cf_last)
        C.Cf_cond[i]=np.linalg.cond(Cf_last, p=2)
        C.Cf_det[i]=np.linalg.det(Cf_last)

        if i<C.Nburn:
            eps_record[i]=eps_last
    
    toc=time.process_time()
    if output_path is not None:
        output_buffer.append(f"McFLI MCMC Time: {toc-tic:.2f}s")
        output_buffer.append(f"Acceptance rate A0: {C.n_a1/C.N*100:.2f} pct.")
        output_buffer.append(f"Acceptance rate na: {C.n_a2/C.N*100:.2f} pct.")
        output_buffer.append(f"Acceptance rate x1: {C.n_a3/C.N*100:.2f} pct.")
        with open(output_path, 'w') as f:
            for line in output_buffer:
                f.write(line + '\n')
        print(f"output file saved to {output_path}")
    
    #%%
    return C,jmp

def InitializeMetropolis(D,C,P,R):
    
    
    Delta=CalcDelta(D.nR,D.nt,D.L)
    DeltaA=CalcADelta(D.nR,D.nt)
    B=CalcB(D.nR,D.nt)
    
    C.thetaA0=empty((D.nR,C.N))
    C.thetaA0[:,0]=P.meanA0
    thetauA0=C.thetaA0[:,0]
    
    C.thetana=empty((D.nR,C.N))    
    C.thetana[:,0]=P.meanna
    thetauna=C.thetana[:,0]
    
    C.thetax1=empty((D.nR,C.N))
    C.thetax1[:,0]=P.meanx1
    thetaux1=C.thetax1[:,0]
    
    thetauq=[]
    
    seed([R.Seed])
    
    R.z1=randn(D.nR,C.N*5)
    R.z2=randn(D.nR,C.N)
    R.z3=randn(D.nR,C.N)
    R.u1=rand(C.N,1)
    R.u2=rand(C.N,1)
    R.u3=rand(C.N,1)
    
    return Delta,DeltaA,B,C,thetauA0,thetauna,thetaux1,thetauq,R


