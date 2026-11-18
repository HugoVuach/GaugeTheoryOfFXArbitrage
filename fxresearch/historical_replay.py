"""Cash replay on raw top-of-book packets. Every fill is hypothetical.

No queue priority or broker rejection probability is inferred from quotes.
Unchanged published prices/sizes do not replenish consumed model liquidity.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def quote_at(t,source,destination,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,max_age=1000):
    p=int(pair_map[source,destination]);lo=int(offsets[p]);hi=int(offsets[p+1]);start=lo
    while lo<hi:
        m=(lo+hi)//2
        if times[m]<=t:lo=m+1
        else:hi=m
    k=lo-1
    if k<start or t-times[k]>max_age:return 0.,0.,-1,p,-1
    if base[p]==source:return bid[k],bvol[k],int(versions[k]),p,0
    return 1/ask[k],avol[k]*ask[k],int(versions[k]),p,1

@njit(cache=True)
def price_labels(decision_times,routes,lengths,gross,delays,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,max_age=1000):
    count=len(decision_times);real=np.full((count,len(delays)),np.nan);edge=np.full_like(real,np.nan)
    for e in range(count):
        t=decision_times[e];length=int(lengths[e])
        for d in range(len(delays)):
            gain=0.;maximum=0.;valid=True
            for k in range(length):
                i=int(routes[e,k]);j=int(routes[e,(k+1)%length])
                r0,_,_,_,_=quote_at(t,i,j,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,max_age)
                r,_,_,_,_=quote_at(t+(k+1)*delays[d],i,j,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,max_age)
                if r<=0 or r0<=0:valid=False;break
                gain+=10000*np.log(r);maximum=max(maximum,10000*np.log(r0/r))
            if valid:real[e,d]=gain;edge[e,d]=maximum
    return real,edge

@njit(cache=True)
def fill(t,i,j,amount,fee,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,current,consumed,max_age,partial=False):
    rate,capacity,version,p,side=quote_at(t,i,j,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,max_age)
    if rate<=0 or not np.isfinite(capacity):return 0.,0.,False,False
    if current[p,side]!=version:current[p,side]=version;consumed[p,side]=0.
    free=max(0.,capacity-consumed[p,side])
    if amount>free+1e-7 and not partial:return 0.,0.,False,True
    used=min(amount,free)
    consumed[p,side]+=used
    return used,used*rate*(1-fee/10000),used>=amount-1e-7,True

@njit(cache=True)
def replay_events(decision_times,routes,lengths,gross,episode_ids,max_ages,threshold,fee,delay,radius,age_filter,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,notional=100000.,initial_cash=1000000.,max_age=1000):
    # t, episode, length, pred return bp, pnl, complete, closed, cash,
    # final time, failed leg, failure kind (0 none,1 depth,2 stale), quote fills.
    records=np.full((len(decision_times),15),np.nan);out=0;busy=-1;last_episode=-1;cash=initial_cash;frozen=False
    current=np.full((len(base),2),-1,np.int64);consumed=np.zeros((len(base),2))
    fee_log=-10000*np.log1p(-fee/10000)
    for e in range(len(decision_times)):
        if frozen:break
        t=decision_times[e];L=int(lengths[e]);episode=episode_ids[e]
        if t<busy or episode==last_episode or cash<notional:continue
        nominal=gross[e]-L*fee_log
        if 10000*np.expm1(nominal/10000)<=threshold or nominal<=L*radius:continue
        if age_filter>0 and max_ages[e]>age_filter:continue
        last_episode=episode;amount=notional;currency=0;complete=True;closed=True;failure=0;failed=-1;filled=0;end=t;residual_currency=-1;residual_amount=0.
        for k in range(L):
            target=int(routes[e,(k+1)%L]);end=t+(k+1)*delay
            used,received,full,observed=fill(end,currency,target,amount,fee,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,current,consumed,max_age)
            if not full:complete=False;failure=1 if observed else 2;failed=k;break
            amount=received;currency=target;filled+=1
        if currency!=0:
            end+=delay
            used,received,full,observed=fill(end,currency,0,amount,fee,offsets,times,bid,ask,bvol,avol,versions,pair_map,base,current,consumed,max_age,True)
            if not full:
                closed=False;frozen=True;failure=1 if observed else 2
                residual_currency=currency;residual_amount=amount-used
            amount=received;filled+=int(used>0)
        pnl=amount-notional if closed else np.nan
        if closed:cash+=pnl
        else:cash+=amount-notional  # Known partial USD proceeds remain actual cash.
        busy=end
        records[out,:]=np.array([t,episode,L,10000*np.expm1(nominal/10000),pnl,1. if complete else 0.,1. if closed else 0.,cash,end,failed,failure,filled,residual_currency,residual_amount,amount])
        out+=1
    return records[:out]
