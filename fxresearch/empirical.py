"""Compiled eight-currency diagnostics; inputs are causal raw tick snapshots.

Karp covers the whole available graph. The funded route is a *simple* USD
cycle chosen by subset DP only when the global graph can contain an opportunity.
It is not a proof about arbitrary position-dependent market impact.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def karp_mean(a):
    n=len(a);d=np.full((n+1,n),-np.inf);d[0,:]=0.
    for k in range(1,n+1):
        for j in range(n):
            value=-np.inf
            for i in range(n):
                candidate=d[k-1,i]+a[i,j]
                if candidate>value:value=candidate
            d[k,j]=value
    best=-np.inf
    for j in range(n):
        if not np.isfinite(d[n,j]):continue
        worst=np.inf
        for k in range(n):
            if np.isfinite(d[k,j]):
                value=(d[n,j]-d[k,j])/(n-k)
                if value<worst:worst=value
        if worst>best:best=worst
    return best

@njit(cache=True)
def best_funded_cycle(a):
    n=len(a);states=1<<n
    dp=np.full((states,n),-np.inf);parent=np.full((states,n),-1,np.int8)
    dp[1,0]=0.;best=-np.inf;last=-1;chosen=-1;length=0
    for mask in range(1,states,2):
        size=0
        for bit in range(n):size+=(mask>>bit)&1
        for j in range(n):
            value=dp[mask,j]
            if not np.isfinite(value):continue
            if j!=0 and np.isfinite(a[j,0]):
                mean=(value+a[j,0])/size
                if mean>best:best=mean;last=j;chosen=mask;length=size
            for k in range(1,n):
                if not mask&(1<<k) and np.isfinite(a[j,k]):
                    candidate=value+a[j,k];new_mask=mask|(1<<k)
                    if candidate>dp[new_mask,k]:dp[new_mask,k]=candidate;parent[new_mask,k]=j
    route=np.full(n,-1,np.int8)
    if last<0:return best,route,0
    mask=chosen;j=last
    for pos in range(length-1,0,-1):
        route[pos]=j;old_j=j;j=int(parent[mask,j]);mask^=1<<old_j
    route[0]=0
    return best,route,length

@njit(cache=True)
def triangle_mean(a):
    best=-np.inf;n=len(a)
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                best=max(best,(a[i,j]+a[j,k]+a[k,i])/3,(a[i,k]+a[k,j]+a[j,i])/3)
    return best

@njit(cache=True)
def full_hodge(a):
    n=len(a);mid=np.zeros((n,n));theta=np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i!=j:mid[i,j]=(a[i,j]-a[j,i])/2
    for i in range(n):theta[i]=-mid[i,:].sum()/n
    energy=0.;edges=0;curvature=0.
    for i in range(n):
        for j in range(i+1,n):
            residual=mid[i,j]-(theta[j]-theta[i]);energy+=residual**2;edges+=1
            for k in range(j+1,n):curvature=max(curvature,abs(mid[i,j]+mid[j,k]+mid[k,i]))
    return np.sqrt(energy/edges),curvature

@njit(cache=True)
def scan_snapshots(bids,asks,ages,base,quote,age_limit=1000.,fee_bp=.1):
    count,pairs=bids.shape;n=8
    # mu (all), mu (100ms), best triangle, full-network Hodge RMS,
    # curvature, available pairs, full-network flag, funded mean, route length.
    metrics=np.full((count,9),np.nan)
    routes=np.full((count,n),-1,np.int8)
    fee=-10000*np.log1p(-fee_bp/10000)
    for t in range(count):
        a=np.full((n,n),-np.inf);fresh100=np.full((n,n),-np.inf)
        available=0
        for p in range(pairs):
            if ages[t,p]>age_limit or bids[t,p]<=0 or asks[t,p]<bids[t,p]:continue
            i=base[p];j=quote[p];a[i,j]=10000*np.log(bids[t,p]);a[j,i]=-10000*np.log(asks[t,p]);available+=1
        # A gauge transformation reduces cancellation from large currency levels.
        phi=np.zeros(n)
        for i in range(1,n):
            if np.isfinite(a[0,i]) and np.isfinite(a[i,0]):phi[i]=(a[0,i]-a[i,0])/2
        for i in range(n):
            for j in range(n):
                if i!=j and np.isfinite(a[i,j]):a[i,j]+=phi[i]-phi[j]
        for p in range(pairs):
            if ages[t,p]<=100:
                i=base[p];j=quote[p];fresh100[i,j]=a[i,j];fresh100[j,i]=a[j,i]
        mu=karp_mean(a);metrics[t,0]=mu;metrics[t,1]=karp_mean(fresh100)
        metrics[t,2]=triangle_mean(a);metrics[t,5]=available;metrics[t,6]=1. if available==28 else 0.
        if available==28:
            rms,curv=full_hodge(a);metrics[t,3]=rms;metrics[t,4]=curv
        if mu>fee+1e-7:
            value,route,length=best_funded_cycle(a)
            if value>fee+1e-7:
                routes[t]=route;metrics[t,7]=value-fee;metrics[t,8]=length
    return metrics,routes
