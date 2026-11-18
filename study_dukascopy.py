"""Frozen full-day study of eight currencies on original Dukascopy packets."""
import argparse,csv,hashlib,json,math,time
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from collect_dukascopy import PAIRS,CURRENCIES,decode
from fxresearch.empirical import scan_snapshots
from fxresearch.historical_replay import price_labels,replay_events
from fxresearch.certification import conditional_risk_upper

PROTOCOL={
 'source':'Dukascopy public daily raw bi5; top-of-book quotes, not fills',
 'currencies':CURRENCIES,'pairs':PAIRS,'start':'2024-01-01','end_exclusive':'2025-01-01',
 'step_ms':100,'max_age_ms':1000,'fee_bp_main':.1,'latencies_ms':[50,100,500],
 'notional_usd':100000,'initial_daily_cash_usd':1000000,'day_end_purge_ms':10000,
 'training':['2024-01-01','2024-04-01'],'calibration':['2024-04-01','2024-10-01'],
 'test':['2024-10-01','2025-01-01'],'alpha':.05,'delta':.05,
 'thresholds_simple_bp':[0,.1,.25,.5,1,2,4,8],
 'receiver_timestamp':'not supplied; source-time causal reconstruction assumption',
 'replenishment':'published economic version changes (bid,ask,volumes); not every duplicate timestamp',
 'order_model':'hypothetical sequential FOK, one cleanup IOC; unclosed inventory freezes the daily account',
 'risk_target':'at least one strictly negative cash PnL OR unresolved inventory in a traded UTC day',
 'candidate_rule':'maximum-mean simple USD cycle on each positive 100ms snapshot; fixed before labels',
 'statistical_status':'exchangeability and iid market days are hypotheses, not verified facts',
}
BASE=np.array([CURRENCIES.index(p[:3]) for p in PAIRS],dtype=np.int64)
QUOTE=np.array([CURRENCIES.index(p[3:]) for p in PAIRS],dtype=np.int64)
MAP=np.full((8,8),-1,np.int64)
for k,(i,j) in enumerate(zip(BASE,QUOTE)):MAP[i,j]=MAP[j,i]=k
DELAYS=np.array(PROTOCOL['latencies_ms'],dtype=np.int64)
# These full dates were already explored in the preceding four-currency paper.
# Keep them in the raw/descriptive corpus, but do not reuse them for calibration.
EXPLORATORY_DATES={f'2024-06-{d:02d}' for d in range(10,15)}|{f'2024-07-{d:02d}' for d in range(8,13)}

def phase(day,source='dukascopy'):
    if source=='dukascopy':
        return 'training' if day<'2024-02-01' else ('calibration' if day<'2024-03-01' else 'test')
    return 'training' if day<'2024-04-01' else ('calibration' if day<'2024-10-01' else 'test')

class DayTape:
    def __init__(self,raw,day,source='dukascopy'):
        self.day=day;self.start=int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
        self.source=source;self.volume_known=source=='dukascopy'
        self.parts=[];self.audit=[];offsets=[0];tt=[];bb=[];aa=[];bv=[];av=[];vv=[]
        for pair in PAIRS:
            if source=='dukascopy':
                p=raw/day/(pair+'.bi5')
                records=decode(p.read_bytes()) if p.exists() else decode(b'')
                original=len(records)
                valid=(records['bid']>0)&(records['ask']>=records['bid'])&np.isfinite(records['bid_volume'])&np.isfinite(records['ask_volume'])&(records['bid_volume']>=0)&(records['ask_volume']>=0)
                records=records[valid]
                divider=1000 if pair.endswith('JPY') else 100000
                t=records['ms'].astype(np.int64)+self.start
                b=records['bid'].astype(float)/divider;a=records['ask'].astype(float)/divider
                bsize=records['bid_volume'].astype(float)*1e6;asize=records['ask_volume'].astype(float)*1e6
            else:
                prev=(date.fromisoformat(day).replace(day=1)-timedelta(days=1)).strftime('%Y%m')
                months=[prev,day[:7].replace('-','')];chunks=[]
                for month in months:
                    p=raw/month/pair/(day+'.npz')
                    if p.exists():
                        with np.load(p) as z:chunks.append((z['times'],z['bid'],z['ask']))
                t,b,a=[np.concatenate([c[k] for c in chunks]) if chunks else np.empty(0) for k in range(3)]
                t=t.astype(np.int64);original=len(t);valid=np.isfinite(b)&np.isfinite(a)&(b>0)&(a>=b)
                t,b,a=t[valid],b[valid],a[valid];order=np.argsort(t,kind='stable');t,b,a=t[order],b[order],a[order]
                bsize=np.full(len(t),np.nan);asize=bsize.copy()
            change=np.ones(len(t),bool)
            if len(t)>1:
                change[1:]=(b[1:]!=b[:-1])|(a[1:]!=a[:-1])
                if self.volume_known:change[1:]|=(bsize[1:]!=bsize[:-1])|(asize[1:]!=asize[:-1])
            version=np.cumsum(change,dtype=np.int64)
            self.parts.append((t,b,a,bsize,asize,version))
            tt.append(t);bb.append(b);aa.append(a);bv.append(bsize);av.append(asize);vv.append(version);offsets.append(offsets[-1]+len(t))
            self.audit.append({'date':day,'pair':pair,'raw_ticks':original,'valid_ticks':len(t),'invalid':original-len(t),
                'equal_time_rows':int((np.diff(t)==0).sum()),'whole_second_rows':int((t%1000==0).sum()),
                'nonpositive_size_rows':int(((bsize<=0)|(asize<=0)).sum()),'unknown_size_rows':0 if self.volume_known else len(t),'economic_versions':int(version[-1]) if len(t) else 0})
        self.flat=(np.array(offsets,np.int64),np.concatenate(tt),np.concatenate(bb),np.concatenate(aa),np.concatenate(bv),np.concatenate(av),np.concatenate(vv),MAP,BASE)

    def snapshot(self,grid):
        bid=np.zeros((len(grid),28));ask=np.zeros_like(bid);age=np.full_like(bid,np.inf)
        for k,(t,b,a,*_) in enumerate(self.parts):
            if not len(t):continue
            index=np.searchsorted(t,grid,side='right')-1;ok=index>=0;safe=np.maximum(index,0)
            bid[ok,k]=b[safe[ok]];ask[ok,k]=a[safe[ok]];age[ok,k]=grid[ok]-t[safe[ok]]
        return bid,ask,age

def ledger_entries(path):
    result={}
    for line in path.read_text().splitlines():
        r=json.loads(line);result[(r['date'],r['pair'])]=r
    return result

def features(raw,result,limit=None,source='dukascopy'):
    result.mkdir(parents=True,exist_ok=True);folder=result/'days';folder.mkdir(exist_ok=True)
    if source=='dukascopy':
        ledger=ledger_entries(result/'source_ledger.jsonl');days=sorted({d for d,p in ledger if '2024-01-01'<=d<'2024-03-21'})
    else:
        completed={month.name for month in raw.iterdir() if month.is_dir() and all((month/p/'complete.json').exists() for p in PAIRS)}
        days=[];d=date(2024,1,1)
        while d<date(2025,1,1):
            month=d.strftime('%Y%m');previous=(d.replace(day=1)-timedelta(days=1)).strftime('%Y%m')
            if month in completed and (d.day!=1 or previous in completed) and d.weekday()!=5:days.append(str(d))
            d+=timedelta(days=1)
    finished=0
    for day in days:
        marker=folder/(day+'.json')
        if marker.exists():
            previous=json.loads(marker.read_text())
            if previous.get('status')=='no_ticks_recorded':continue
            if all((folder/(day+suffix)).exists() for suffix in ['_events.npz','_sample10s.npz','_hours.csv','_quality.csv']):continue
        if source=='dukascopy' and any((day,p) not in ledger or ledger[day,p]['status'] not in ('ok','empty','absent') for p in PAIRS):continue
        tape=DayTape(raw,day,source)
        if not sum(len(p[0]) for p in tape.parts):
            (folder/(day+'.json')).write_text(json.dumps({'date':day,'status':'no_ticks_recorded','phase':phase(day,source)}));continue
        start=time.monotonic();hours=[];samples=[];events=[]
        for hour in range(24):
            grid=tape.start+hour*3600000+np.arange(0,3600000,100,dtype=np.int64)
            b,a,ages=tape.snapshot(grid)
            metrics,routes=scan_snapshots(b,a,ages,BASE,QUOTE,1000.,0.)
            h={'date':day,'phase':phase(day,source),'hour':hour,'grid':len(grid),'any_pair_points':int((metrics[:,5]>0).sum()),
               'full_28_fresh':int(metrics[:,6].sum()),'available_pair_sum':float(metrics[:,5].sum()),
               'global_max_mean_bp':float(np.nanmax(metrics[:,0])) if np.isfinite(metrics[:,0]).any() else None,
               'hodge_count':int(np.isfinite(metrics[:,3]).sum()),'hodge_sum':float(np.nansum(metrics[:,3])),
               'hodge_max':float(np.nanmax(metrics[:,3])) if np.isfinite(metrics[:,3]).any() else None}
            for fee in [0,.1,.2,.5,1.]:
                cost=-10000*np.log1p(-fee/10000);tag=str(fee)
                h['positive_'+tag]=int((metrics[:,0]>cost+1e-7).sum())
                h['positive_age100_'+tag]=int((metrics[:,1]>cost+1e-7).sum())
                h['positive_triangle_'+tag]=int((metrics[:,2]>cost+1e-7).sum())
                h['positive_no_triangle_'+tag]=int(((metrics[:,0]>cost+1e-7)&(metrics[:,2]<=cost+1e-7)).sum())
            idx=np.flatnonzero(np.isfinite(metrics[:,7])&(grid<tape.start+86400000-PROTOCOL['day_end_purge_ms']))
            if len(idx):
                maxages=np.zeros(len(idx))
                for row,j in enumerate(idx):
                    length=int(metrics[j,8]);route=routes[j,:length]
                    maxages[row]=max(ages[j,MAP[route[k],route[(k+1)%length]]] for k in range(length))
                real,edge=price_labels(grid[idx],routes[idx],metrics[idx,8].astype(np.int64),metrics[idx,7]*metrics[idx,8],DELAYS,*tape.flat)
                quoted,quoted_edge=price_labels(grid[idx],routes[idx],metrics[idx,8].astype(np.int64),metrics[idx,7]*metrics[idx,8],DELAYS,*tape.flat,np.inf)
                events.append((grid[idx],routes[idx],metrics[idx,8].astype(np.int64),metrics[idx,7]*metrics[idx,8],maxages,real,edge,quoted,quoted_edge))
            samples.append(np.column_stack((grid[::100],metrics[::100])))
            hours.append(h)
        if events:
            fields=[np.concatenate([e[k] for e in events]) for k in range(9)]
            t,r,L,g,age,real,edge,quoted,quoted_edge=fields
            new=np.r_[True,(np.diff(t)>100)|np.any(r[1:]!=r[:-1],axis=1)]
            episode=np.cumsum(new,dtype=np.int64)
        else:
            t=np.empty(0,np.int64);r=np.empty((0,8),np.int8);L=np.empty(0,np.int64);g=np.empty(0);age=g.copy();real=np.empty((0,3));edge=real.copy();episode=L.copy()
            quoted=real.copy();quoted_edge=edge.copy()
        np.savez_compressed(folder/(day+'_events.npz'),times=t,routes=r,lengths=L,gross=g,max_ages=age,realized=real,edge=edge,archived_realized=quoted,archived_edge=quoted_edge,episodes=episode)
        np.savez_compressed(folder/(day+'_sample10s.npz'),sample=np.concatenate(samples))
        pd.DataFrame(hours).to_csv(folder/(day+'_hours.csv'),index=False)
        pd.DataFrame(tape.audit).to_csv(folder/(day+'_quality.csv'),index=False)
        scores={}
        for k,delay in enumerate(DELAYS):
            observed=np.isfinite(real[:,k]);errors=np.full(len(t),np.inf);errors[observed]=np.maximum(0,(g[observed]-real[observed,k])/L[observed])
            edge_errors=np.where(np.isfinite(edge[:,k]),edge[:,k],np.inf)
            archived=np.maximum(0,(g-quoted[:,k])/L) if len(t) else np.empty(0)
            scores[str(delay)]={'cycle_mean_error':float(errors.max()) if len(t) else 0.,'edge_error':float(edge_errors.max()) if len(t) else 0.,'label_missing':int((~observed).sum()),'archived_cycle_mean_error':float(archived.max()) if len(t) else 0.,'archived_edge_error':float(quoted_edge[:,k].max()) if len(t) else 0.}
        daily={'date':day,'phase':phase(day,source),'source':source,'status':'processed','raw_ticks':sum(a['raw_ticks'] for a in tape.audit),'candidate_frames':len(t),'candidate_episodes':int(episode[-1]) if len(episode) else 0,'scores':scores,'seconds':time.monotonic()-start}
        (folder/(day+'.json')).write_text(json.dumps(daily,indent=2));print(json.dumps(daily),flush=True)
        finished+=1
        if limit and finished>=limit:break

def upper_quantile(scores,alpha=.05):
    """Exact rank; +inf upper scores represent unobservable mature labels."""
    values=np.asarray(scores,float)
    if not len(values) or np.isnan(values).any() or np.isneginf(values).any():raise ValueError('nonempty upper-score vector required')
    from decimal import Decimal,ROUND_CEILING
    k=int((Decimal(len(values)+1)*(1-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return float(np.partition(values,k-1)[k-1]) if k<=len(values) else np.inf

def augment_archived(raw,result,source='dukascopy'):
    """Add the distinct last-archived-price target to existing feature caches."""
    folder=result/'days'
    for path in sorted(folder.glob('????-??-??.json')):
        metadata=json.loads(path.read_text())
        if metadata.get('status')!='processed':continue
        packet=folder/(metadata['date']+'_events.npz')
        with np.load(packet) as z:arrays={k:z[k] for k in z.files}
        if 'archived_realized' in arrays:continue
        tape=DayTape(raw,metadata['date'],source)
        real,edge=price_labels(arrays['times'],arrays['routes'],arrays['lengths'],arrays['gross'],DELAYS,*tape.flat,np.inf)
        arrays['archived_realized']=real;arrays['archived_edge']=edge
        np.savez_compressed(packet,**arrays)
        for k,delay in enumerate(DELAYS):
            errors=np.maximum(0,(arrays['gross']-real[:,k])/arrays['lengths'])
            metadata['scores'][str(delay)].update(archived_cycle_mean_error=float(errors.max()) if len(errors) else 0.,archived_edge_error=float(edge[:,k].max()) if len(errors) else 0.)
        path.write_text(json.dumps(metadata,indent=2));print('archived price labels',metadata['date'],flush=True)

def calibration(result,alpha=.05,save=True):
    folder=result/'days';days=[json.loads(p.read_text()) for p in sorted(folder.glob('????-??-??.json'))]
    cal=[d for d in days if d.get('status')=='processed' and d['phase']=='calibration' and date.fromisoformat(d['date']).weekday()<5 and d['date'] not in EXPLORATORY_DATES]
    rows=[]
    for delay in DELAYS:
        for method,key in [('joint_block','cycle_mean_error'),('edge_block','edge_error'),('joint_archived','archived_cycle_mean_error'),('edge_archived','archived_edge_error')]:
            values=[d['scores'][str(delay)][key] for d in cal]
            q=upper_quantile(values,alpha)
            rows.append({'delay_ms':int(delay),'method':method,'calibration_days':len(cal),'radius_bp':q,'unknown_day_count':sum(not np.isfinite(v) for v in values)})
        errors=[]
        for d in cal:
            e=np.load(folder/(d['date']+'_events.npz'));k=list(DELAYS).index(delay)
            observed=np.isfinite(e['realized'][:,k]);err=np.full(len(observed),np.inf)
            err[observed]=np.maximum(0,(e['gross'][observed]-e['realized'][observed,k])/e['lengths'][observed]);errors.append(err)
        all_errors=np.concatenate(errors) if errors else np.array([])
        q=upper_quantile(all_errors,alpha) if len(all_errors) else np.inf
        rows.append({'delay_ms':int(delay),'method':'pooled_points','calibration_days':len(cal),'radius_bp':q,'unknown_day_count':sum(d['scores'][str(delay)]['label_missing']>0 for d in cal)})
        errors=[]
        for d in cal:
            e=np.load(folder/(d['date']+'_events.npz'));k=list(DELAYS).index(delay)
            errors.append(np.maximum(0,(e['gross']-e['archived_realized'][:,k])/e['lengths']))
        q=upper_quantile(np.concatenate(errors),alpha) if errors and sum(map(len,errors)) else np.inf
        rows.append({'delay_ms':int(delay),'method':'pooled_archived','calibration_days':len(cal),'radius_bp':q,'unknown_day_count':0})
    for r in rows:r['alpha']=alpha
    if save:pd.DataFrame(rows).to_csv(result/'calibration.csv',index=False)
    return rows

def replay(raw,result):
    radii=calibration(result,save=False);radii={(r['delay_ms'],r['method']):r['radius_bp'] for r in radii}
    folder=result/'days';output=result/'replay';output.mkdir(exist_ok=True)
    methods=[]
    for value in PROTOCOL['thresholds_simple_bp']:methods.append((f'threshold_{value:g}',value,.1,100,0.,0))
    for fee in [0,.2,.5]:methods.append((f'fee_{fee:g}',0,fee,100,0.,0))
    for delay in [50,500]:methods.append((f'latency_{delay}',0,.1,delay,0.,0))
    methods.append(('age_100ms',0,.1,100,0.,100))
    for fee in [.2,.5]:methods.append((f'age100_fee_{fee:g}',0,fee,100,0.,100))
    for delay in [50,500]:methods.append((f'age100_latency_{delay}',0,.1,delay,0.,100))
    for name in ['notional_10k','notional_1m']:methods.append((name,0,.1,100,0.,0))
    for method in ['joint_block','edge_block','pooled_points','joint_archived','edge_archived','pooled_archived']:methods.append((method,0,.1,100,radii[100,method],0))
    columns=['timestamp_ms','episode','length','predicted_simple_bp','pnl_usd','complete','closed','cash_usd','end_ms','failed_leg','failure_kind','filled_legs','residual_currency_index','residual_amount','terminal_usd_received']
    all_daily=[]
    for metadata in sorted(folder.glob('????-??-??.json')):
        day=json.loads(metadata.read_text())
        if day.get('status')!='processed':continue
        d=day['date'];target=output/(d+'_daily.csv')
        cached_rows={}
        if target.exists():
            cached=pd.read_csv(target)
            if 'replay_schema_version' in cached:
                cached_rows={r['method']:r for r in cached[cached.replay_schema_version==2].to_dict('records')}
            if set(cached_rows)=={m[0] for m in methods}:
                all_daily.extend(cached_rows.values());continue
        tape=DayTape(raw,d);e=np.load(folder/(d+'_events.npz'));rows=[]
        for name,threshold,fee,delay,radius,agefilter in methods:
            if name in cached_rows:
                rows.append(cached_rows[name]);continue
            notional=10000. if name=='notional_10k' else (1000000. if name=='notional_1m' else 100000.)
            records=replay_events(e['times'],e['routes'],e['lengths'],e['gross'],e['episodes'],e['max_ages'],threshold,fee,delay,radius,agefilter,*tape.flat,notional,10*notional)
            closed=records[:,6].astype(bool);pnl=records[:,4];unknown=int((~closed).sum());losses=int((pnl[closed]<-1e-8).sum())
            row={'date':d,'phase':day['phase'],'weekday':date.fromisoformat(d).weekday(),'method':name,'accepted':len(records),'losses':losses,'unclosed':unknown,'complete':int(records[:,5].sum()),'known_pnl_usd':float(np.nansum(pnl)),
                 'conservative_pnl_usd':float(np.nansum(pnl)+np.sum(records[~closed,14]-notional)),'any_loss':int(losses>0 or unknown>0),'accepted_day':int(bool(len(records))),'radius_bp':radius,'fee_bp':fee,'delay_ms':delay,'notional_usd':notional,'replay_schema_version':2}
            rows.append(row)
            if len(records) and abs(row['conservative_pnl_usd']-(records[-1,7]-10*notional))>1e-5:
                raise AssertionError('cash ledger and conservative terminal value disagree')
            if len(records):
                frame=pd.DataFrame(records,columns=columns)
                frame['date']=d;frame['method']=name;frame['phase']=day['phase'];frame['hypothetical']=True
                frame['residual_currency']=[CURRENCIES[int(i)] if i>=0 else '' for i in frame.residual_currency_index]
                indexes=np.searchsorted(e['times'],frame.timestamp_ms.to_numpy(dtype=np.int64))
                frame['route_json']=[json.dumps([CURRENCIES[int(v)] for v in e['routes'][idx,:e['lengths'][idx]]]+['USD']) for idx in indexes]
                frame.to_csv(output/(d+'_'+name+'_attempts.csv.gz'),index=False,compression='gzip')
        pd.DataFrame(rows).to_csv(target,index=False);all_daily.extend(rows)
        print('replay',d,'candidates',len(e['times']),flush=True)
    daily=pd.DataFrame(all_daily);daily.to_csv(result/'daily_policies.csv',index=False)
    cal=daily[(daily.phase=='calibration')&(daily.weekday<5)]
    validation=[]
    for threshold in PROTOCOL['thresholds_simple_bp']:
        name=f'threshold_{threshold:g}';sub=cal[cal.method==name];n=int(sub.accepted_day.sum());k=int(sub.any_loss.sum())
        u=conditional_risk_upper(k,n,.05,8)
        validation.append({'method':name,'traded_calibration_days':n,'losing_or_unknown_days':k,'risk_upper':u,'validated':u<=.05,'calibration_pnl':sub.conservative_pnl_usd.sum()})
    good=[v for v in validation if v['validated']];chosen=max(good,key=lambda v:v['calibration_pnl'])['method'] if good else None
    (result/'policy_validation.json').write_text(json.dumps({'policies':validation,'selected_policy':chosen,'family_size':8,'assumptions':PROTOCOL['statistical_status'],'target':PROTOCOL['risk_target']},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','augment','calibrate','replay'])
    p.add_argument('--raw',type=Path,default=Path('raw/dukascopy'));p.add_argument('--result',type=Path,default=Path('results/dukascopy'));p.add_argument('--limit',type=int);p.add_argument('--source',choices=['dukascopy','histdata'],default='dukascopy')
    args=p.parse_args()
    if args.stage=='features':features(args.raw,args.result,args.limit,args.source)
    elif args.stage=='augment':augment_archived(args.raw,args.result,args.source)
    elif args.stage=='calibrate':
        rows=[r for alpha in [.1,.05,.01] for r in calibration(args.result,alpha,False)]
        pd.DataFrame(rows).to_csv(args.result/'calibration.csv',index=False)
        print(json.dumps(rows),flush=True)
    else:
        if args.source!='dukascopy':raise ValueError('HistData sizes are unknown; no capacity-based cash replay is justified')
        replay(args.raw,args.result)

if __name__=='__main__':main()
