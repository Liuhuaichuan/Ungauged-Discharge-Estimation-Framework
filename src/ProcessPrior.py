#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#这个是用全时段的观测值来计算全时段的流量，从而找到最佳先验
from numpy import empty,ones,zeros,mean,std,median,exp,maximum
import numpy as np
from numpy.random import randn,rand,seed
from scipy.stats import lognorm
import time
from logninvstat import logninvstat
from calcnhat import calcnhat
from MetroManVariables import Jump

def replace_invalid_in_column(column_vector, condition, window_size=2):

    result = column_vector.copy()
    
    invalid_indices = np.where(~condition)[0]
    
    for idx in invalid_indices:
        start = max(0, idx - window_size // 2)
        end = min(len(column_vector), idx + window_size // 2 + 1)
        
        window = column_vector[start:end]
        valid_values = window[condition[start:end]]
        
        if len(valid_values) > 0:
            result[idx] = np.mean(valid_values)
        else:
            expanded_window_size = window_size * 2
            expanded_start = max(0, idx - expanded_window_size // 2)
            expanded_end = min(len(column_vector), idx + expanded_window_size // 2 + 1)
            
            expanded_window = column_vector[expanded_start:expanded_end]
            expanded_valid_values = expanded_window[condition[expanded_start:expanded_end]]
            
            if len(expanded_valid_values) > 0:
                result[idx] = np.mean(expanded_valid_values)
            else:
                global_valid_values = column_vector[condition]
                if len(global_valid_values) > 0:
                    result[idx] = np.mean(global_valid_values)
                else:
                    result[idx] = 0
    
    return result

def ProcessPrior(Prior,AllObs,DAll,Obs,D,ShowFigs,E,R,SoS_params,DebugMode,Use_sos):
    SoS_A0=replace_invalid_in_column(SoS_params.sosA0, SoS_params.sosA0 > 0)
    SoS_na=replace_invalid_in_column(SoS_params.sosna, SoS_params.sosna > 0)
    SoS_x1=replace_invalid_in_column(SoS_params.sosx1, SoS_params.sosx1 > -100)
    #%% 1 handle input prior information
    #% note that A0min is refined for inclusion in the "jmp" variable at the bottom
    allA0min=empty((DAll.nR,1))
    for i in range(0,DAll.nR):
        if min(AllObs.dA[i,:] ) >= 0:
            allA0min[i,0]=1e-3
        else:
            allA0min[i,0]=-min(AllObs.dA[i,:])+1e-3
          
    Obs.hmin=Obs.h.min(1)
    AllObs.hmin=AllObs.h.min(1)
    if Use_sos:
        A0u=SoS_A0
    else:
        A0u=ones((DAll.nR,1))*0.27*(Prior.meanQbar**.39)*7.2*(Prior.meanQbar**0.5) #Moody & Troutman A0 ，计算初始河道横截面积（initial cross-sectional area）的估计值   
        
    for r in range(0,D.nR):
        if A0u[r]<allA0min[r]:
            A0u[r]=allA0min[r]+1
    #%% 2 friction coefficient， na
    meanx1=empty((D.nR,1))
    meanna=empty((D.nR,1))    
    for r in range(0,DAll.nR):
        if E.nOpt==3:
            if Use_sos:
                meanna[r]=SoS_na[r]
            else:
                meanna[r]=0.04
            covna=0.05
        elif E.nOpt==4:
            if Use_sos:
                meanna[r]=SoS_na[r]
            else:
                meanna[r]=0.04
            covna=0.05
        elif E.nOpt==5:
            if Use_sos:
                meanna[r]=SoS_na[r]
            else:
                meanna[r]=0.03
            covna=0.05

    for r in range(0,DAll.nR):
        if E.nOpt==3:
            meanx1[r]=-0.1
            covx1=0.25
        elif E.nOpt==4:
            meanx1[r]=-0.25
            covx1=1
        elif E.nOpt==5:
            if Use_sos:
                meanx1[r]=np.abs(SoS_x1[r])
            else:
                covd=0.3; #Moody and troutman
                meanx1[r]=A0u[r]/mean(AllObs.w[r,:])*covd
            covx1=0.5

       
    #%% 3 initial probability calculations
    v=(covna*meanna)**2
    [mun,sigman]=logninvstat(meanna,v)
    
    v=(covx1*meanx1)**2
    [mux1,sigmax1] = logninvstat(np.abs(meanx1),v)
    
    v=(Prior.covQbar*Prior.meanQbar)**2
    [muQbar,sigmaQbar] = logninvstat(Prior.meanQbar,v) 

    v=(0.1*A0u)**2
    [muA0,sigmaA0] = logninvstat(A0u,v)  
    
    #%%  chain setup
    N=int(1e4) #10000
    
    if DebugMode:
        N=int(1e3)
    
    Nburn=int(N*.2)#2000

    nau=meanna
    x1u=meanx1
    
    seed([R.Seed])
    
    z1=randn(DAll.nR,N)
    z2=randn(DAll.nR,N)
    z3=randn(DAll.nR,N)
    u1=rand(DAll.nR,N)
    u2=rand(DAll.nR,N)
    u3=rand(DAll.nR,N)
    na1=zeros((D.nR,1))
    na2=zeros((D.nR,1))
    na3=zeros((D.nR,1))
    
    thetaAllA0=empty( (DAll.nR,N) )
    for r in range(0,DAll.nR):
        thetaAllA0[r,0]=A0u[r]
    thetana=empty( (DAll.nR,N) )
    for r in range(0,DAll.nR):
        thetana[r,0]=nau[r]
    thetax1=empty( (DAll.nR,N))
    for r in range(0,DAll.nR):
        thetax1[r,0]=x1u[r]
    thetaQ=empty( (DAll.nR,N))
    f=empty( (DAll.nR,N))
    
    jstdA0s=empty( (D.nR,N) )
    jstdnas=empty( (D.nR,N) )
    jstdx1s=empty( (D.nR,N) )
    
    #%% chain calculations
    tic=time.process_time()
    for j in range(0,DAll.nR):
    # for j in range(0,1):
        

        print("Processing prior for reach",j+1,"/",D.nR,".")
        
        A0u=thetaAllA0[j,0]
        nau=thetana[j,0]
        x1u=thetax1[j,0]
        
        jstdA0=A0u
        jstdna=nau
        jstdx1=0.1*x1u
        
        jtarget=0.5
        
        Au=A0u+AllObs.dA[j,:]
        Abaru=median(Au)
        
        if Prior.Geomorph.Use:
            pu1A=lognorm.pdf(Abaru,Prior.Geomorph.logA0_sigma,0,exp(Prior.Geomorph.logA0_hat))
        else:
            pu1A=1
        pu1=lognorm.pdf(A0u,sigmaA0[j],0,exp(muA0[j]))
        pu2=lognorm.pdf(nau,sigman[j],0,exp(mun[j]))
        if E.nOpt<5:
            pu3=lognorm.pdf(-x1u,sigmax1[j],0,exp(mux1[j]) )
        elif E.nOpt==5:
            pu3=lognorm.pdf(x1u,sigmax1[j],0,exp(mux1[j]) )
        
        nhatu = calcnhat(AllObs.w[j,:],AllObs.h[j,:],AllObs.hmin[j],A0u+AllObs.dA[j,:],x1u,nau,E.nOpt)
        
        Qu = mean( 1/nhatu * (Au)**(5/3) * AllObs.w[j,:]**(-2/3)* AllObs.S[j,:]**0.5 )
        
        fu = 1#lognorm.pdf(Qu,sigmaQbar,0,exp(muQbar) )
        
        for i in range(0,N):
            
            if i%500==0:
                print(f'Reach {j+1}, Iteration {i}')
            #adaptation
            if i<N*0.2 and i>0 and i%100==0:
                jstdA0=mean(jstdA0s[j,0:i-1] )/jtarget*(na1[j]/i)
                jstdna=mean(jstdnas[j,0:i-1] )/jtarget*(na2[j]/i)
                jstdx1=mean(jstdx1s[j,0:i-1] )/jtarget*(na3[j]/i)

            jstdA0s[j,i]=jstdA0 #this part is very messy
            jstdnas[j,i]=jstdna
            jstdx1s[j,i]=jstdx1
            
            #A0
            A0v=A0u+z1[j,i]*jstdA0
            Av=A0v+AllObs.dA[j,:]
            Abarv=median(Av)
                        
            if A0v<allA0min[j]:
                pv1=0; fv=0; pv1A=0;
            else:
                #pv1=1
                pv1=lognorm.pdf(A0v,sigmaA0[j],0,exp(muA0[j]))
                Qv=mean( 1/nhatu * (Av)**(5/3) * AllObs.w[j,:]**(-2/3)* AllObs.S[j,:]**0.5 )
                #fv = lognorm.pdf(Qv,sigmaQbar,0,exp(muQbar) )
                fv=1
                if Prior.Geomorph.Use:
                    pv1A=lognorm.pdf(Abarv,Prior.Geomorph.logA0_sigma,0,exp(Prior.Geomorph.logA0_hat))
                else:
                    pv1A=1
            
            MetRatio=fv/fu*pv1/pu1*pv1A/pu1A
            
            if MetRatio > u1[j,i]:
                na1[j]=na1[j]+1 # Update current state and increment acceptance count na1[j]
                A0u=A0v; Au=Av; Qu=Qv; 
                fu=fv; pu1=pv1; pu1A=pv1A;
            
            #na
            nav=nau+z2[j,i]*jstdna
            if (nav <= 0) | (nav > 1):
                pv2=0
            else:
                pv2=lognorm.pdf(nav,sigman[j],0,exp(mun[j]))
            
            nhatv = calcnhat(AllObs.w[j,:],AllObs.h[j,:],AllObs.hmin[j],A0u+AllObs.dA[j,:],x1u,nav,E.nOpt)
            Qv=mean( 1/nhatv * (Au)**(5/3) * AllObs.w[j,:]**(-2/3)* AllObs.S[j,:]**0.5 ) 
            #fv=lognorm.pdf(Qv,sigmaQbar,0,exp(muQbar) )
            fv=1
            
            MetRatio=fv/fu*pv2/pu2
            
            if MetRatio >u2[j,i]:
                na2[j]=na2[j]+1
                nau=nav; Qu=Qv;
                fu=fv; pu2=pv2;
                
            #x1
            x1v=x1u+z3[j,i]*jstdx1# x1v < 0 if E.nOpt < 5 else > 0
            if E.nOpt<5:
                if x1v >=0:
                    pv3=0
                else:
                    pv3=lognorm.pdf(-x1v,sigmax1[j],0,exp(mux1[j]) )
            elif E.nOpt==5:
                if x1v <0:
                    pv3=0
                else:
                    pv3=lognorm.pdf(x1v,sigmax1[j],0,exp(mux1[j]) )
                
            nhatv = calcnhat(AllObs.w[j,:],AllObs.h[j,:],AllObs.hmin[j],A0u+AllObs.dA[j,:],x1v,nau,E.nOpt)
            Qv=mean( 1/nhatv * (Au)**(5/3) * AllObs.w[j,:]**(-2/3)* AllObs.S[j,:]**0.5 ) 
            #fv=lognorm.pdf(Qv,sigmaQbar,0,exp(muQbar) )
            fv=1
            MetRatio=fv/fu*pv3/pu3
            
            if MetRatio >u3[j,i]:
                na3[j]=na3[j]+1
                x1u=x1v; Qu=Qv;
                fu=fv; pu3=pv3;
            
            
            thetaAllA0[j,i]=A0u
            thetana[j,i]=nau
            thetax1[j,i]=x1u
            thetaQ[j,i]=Qu
            f[j,i]=fu

    toc=time.process_time(); print('Prior MCMC Time: %.2fs' %(toc-tic))    
                   
    #%% 4. Calculating final prior parameters
    Prior.meanAllA0=mean(thetaAllA0[:,Nburn+1:N],axis=1)
    Prior.stdAllA0=std(thetaAllA0[:,Nburn+1:N],axis=1 )
    Prior.meanna=mean(thetana[:,Nburn+1:N],axis=1  )
    Prior.stdna=std(thetana[:,Nburn+1:N],axis=1  )
    Prior.meanx1=mean(thetax1[:,Nburn+1:N],axis=1  )
    Prior.stdx1=std(thetax1[:,Nburn+1:N],axis=1  )
    
    #%% 5. calculate minimum values for A0 for the estimation window
    #5.1 calculate minimum values for A0 for the estimation window
    estA0min=empty( ( D.nR,1))
    for i in range(0,D.nR):
        if min(Obs.dA[i,:] ) >=0:
            estA0min[i,:]=0
        else:
            estA0min[i,:]=-min(Obs.dA[i,:] )
    #5.3 shift the "all" A0 into the estimate window
    AllObs.A0Shift=AllObs.dA[:,E.iEst[0]] #different than the Matlab version... should be ok?
    
    #5.4 save the more restrictive limit
    Amin=1; #this is the lowest value that we will let A0+dA take
    jmp=Jump()
    jmp.A0min=maximum(allA0min.T+AllObs.A0Shift,estA0min.T)+Amin
    jmp.nmin=0.001
    
    #5.5 set up prior A0 variable by shifting into estimation window
    Prior.meanA0=Prior.meanAllA0+AllObs.A0Shift
    Prior.stdA0=Prior.stdAllA0
    
    return Prior,jmp
