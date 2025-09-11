from numpy import reshape,empty,ones,any,concatenate,isinf
from numpy.linalg import inv,cond
import numpy as np
#from ManningCalcs import *
# from ChannelMassBal import *
from calcnhat import calcnhat

def CalcLklhd(Obs,AllObs,A0,na,x1,D,Prior,Delta,DeltaA,B,qhatv,nOpt,eps_value=None):

    # All vectors ordered "space-first"
    # theta(1)=theta(r1,t1)
    # theta(2)=theta(r1,t2)
    # ... 
    # theta(nt)=theta(r1,nt)
    # theta(nt+1)=theta(r2,t1)

    # prep
    M=D.nR * D.nt
    N=D.nR *(D.nt-1)

    A0=A0.reshape(len(A0),1) #blech... surely there's a better way
    
    A0v=(A0*ones([1,D.nt])).reshape(D.nR*D.nt,1) #seems silly...
    
    nhat=empty((D.nR,D.nt))
    for r in range(0,D.nR):
        nhat[r,:]=calcnhat(Obs.w[r,:],Obs.h[r,:],Obs.hmin[r],A0[r]+Obs.dA[r,:],x1[r],na[r],nOpt)
    
    nv=reshape(nhat.T,(M,1),order='F') #setting order to F makes it Matlab-equivalent I think
    Qv=1/nv*(A0v+Obs.dAv)**(5/3)*Obs.wv**(-2/3)*Obs.Sv**(1/2)
    

    if (A0v<0).any() | (Obs.Sv<0).any():
        f=0
        print(Obs.hv, A0v, Obs.Sv)
        return f
    
    #%%1) Calculate dQdx, dQdt, and q for channel mass balance
    dQdxv=Delta @ Qv
    dAdtv=(DeltaA @ Obs.hv) / D.dt * (B @ Obs.wv) 
    
    #%%2) Calculate covariance matrix of theta
    #2.1) Calculate covariance matrix of dQdx
    
    TSv=Obs.Sv**(-1)
    TdAv=1/(A0v+Obs.dAv)
    Tw=Obs.wv**(-1)    
    JS=0.5*Delta* (ones((N,1))@Qv.T)*(ones((N,1)) @ TSv.T)#ones((N,1))@Qv.T
    JdA=5/3*Delta*(ones((N,1))@Qv.T)*(ones((N,1)) @ TdAv.T)
    Jw=-2/3*Delta*(ones((N,1))@Qv.T)*(ones((N,1)) @ Tw.T)
    
    J=concatenate((JS,JdA,Jw),axis=1 )#N*3M
    CdQ=J @ Obs.CSdAw @ J.T#N*N，dQdx

    Cf=Obs.CA + CdQ + Prior.Cqf
    eps_value=1e-8
    Cf = Cf + np.eye(Cf.shape[0]) * eps_value
    
    Theta=dQdxv+dAdtv-Prior.Lats.qv
    
    try:
        f = -0.5 * Theta.T @ np.linalg.solve(Cf, Theta)
    except np.linalg.LinAlgError:
        f = 0
        print('f=0')
        
    return f, Theta,Cf,eps_value #N*1
