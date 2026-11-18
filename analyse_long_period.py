"""Summaries, training-only scales and held-out price/cash evaluations."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
from study_dukascopy import calibration,upper_quantile,DELAYS,EXPLORATORY_DATES

AGE_CUTS=np.array([0,100,250,500,1001.])

def scale_index(lengths,ages):
    return np.clip(lengths.astype(int)-2,0,6),np.clip(np.searchsorted(AGE_CUTS,ages,side='right')-1,0,3)

def entries(result):
    return [json.loads(p.read_text()) for p in sorted((result/'days').glob('????-??-??.json'))]

def dataset_summary(result):
    records=[d for d in entries(result) if d.get('status')=='processed']
    hours=pd.concat([pd.read_csv(result/'days'/(d['date']+'_hours.csv')) for d in records],ignore_index=True)
    quality=pd.concat([pd.read_csv(result/'days'/(d['date']+'_quality.csv')) for d in records],ignore_index=True)
    hours.to_csv(result/'hourly.csv',index=False);quality.to_csv(result/'quality.csv',index=False)
    summary={'first_date':min(d['date'] for d in records),'last_date':max(d['date'] for d in records),'dates':len(records),
        'weekday_dates':sum(pd.Timestamp(d['date']).weekday()<5 for d in records),'raw_ticks':int(quality.raw_ticks.sum()),'valid_ticks':int(quality.valid_ticks.sum()),
        'invalid_ticks':int(quality.invalid.sum()),'whole_second_fraction':float(quality.whole_second_rows.sum()/quality.valid_ticks.sum()),
        'duplicate_timestamp_fraction':float(quality.equal_time_rows.sum()/quality.valid_ticks.sum()),'unknown_size_fraction':float(quality.unknown_size_rows.sum()/quality.valid_ticks.sum()),
        'grid_points':int(hours.grid.sum()),'active_grid_points':int(hours.any_pair_points.sum()),'full_28_fresh_points':int(hours.full_28_fresh.sum()),
        'candidate_frames':sum(d['candidate_frames'] for d in records),'candidate_episodes':sum(d['candidate_episodes'] for d in records),
        'phase_weekdays':{phase:sum(d['phase']==phase and pd.Timestamp(d['date']).weekday()<5 and not (phase=='calibration' and d['date'] in EXPLORATORY_DATES) for d in records) for phase in ['training','calibration','test']},
        'previously_explored_dates_excluded_from_calibration':sorted(d['date'] for d in records if d['phase']=='calibration' and d['date'] in EXPLORATORY_DATES),
        'global_max_mean_bp':float(hours.global_max_mean_bp.max()),'hodge_rms_average_full_network':float(hours.hodge_sum.sum()/hours.hodge_count.sum())}
    fees=[]
    for fee in [0,.1,.2,.5,1.]:
        key=str(fee)
        fees.append({'fee_bp':fee,'global_positive_frames':int(hours['positive_'+key].sum()),'triangle_positive_frames':int(hours['positive_triangle_'+key].sum()),
                     'global_positive_age100_frames':int(hours['positive_age100_'+key].sum()),'global_positive_without_triangle_frames':int(hours['positive_no_triangle_'+key].sum()),'active_grid_points':summary['active_grid_points']})
    pd.DataFrame(fees).to_csv(result/'fees.csv',index=False)
    bypair=quality.groupby('pair')[['raw_ticks','valid_ticks','invalid','equal_time_rows','whole_second_rows','nonpositive_size_rows','unknown_size_rows']].sum()
    bypair.to_csv(result/'quality_by_pair.csv')
    pd.DataFrame(records).drop(columns=['scores']).to_csv(result/'daily.csv',index=False)
    (result/'dataset_summary.json').write_text(json.dumps(summary,indent=2));return summary

def fit_training_scales(result):
    train=[d for d in entries(result) if d.get('status')=='processed' and d['phase']=='training' and pd.Timestamp(d['date']).weekday()<5]
    groups={};all_errors=[[] for _ in DELAYS];counts=np.zeros((3,7,4),int)
    for day in train:
        with np.load(result/'days'/(day['date']+'_events.npz')) as e:
            li,ai=scale_index(e['lengths'],e['max_ages'])
            for k in range(3):
                error=np.maximum(0,(e['gross']-e['archived_realized'][:,k])/e['lengths']);all_errors[k].append(error)
                for length in range(7):
                    for age in range(4):
                        values=error[(li==length)&(ai==age)]
                        groups.setdefault((k,length,age),[]).append(values);counts[k,length,age]+=len(values)
    scales=np.zeros((3,7,4))
    for k in range(3):
        all_values=np.concatenate(all_errors[k]);fallback=max(.01,float(np.quantile(all_values,.9,method='higher')))
        for length in range(7):
            for age in range(4):
                values=np.concatenate(groups[k,length,age]);scales[k,length,age]=max(.01,float(np.quantile(values,.9,method='higher'))) if len(values)>=100 else fallback
    model={'training_days':len(train),'age_cutoffs_ms':AGE_CUTS.tolist(),'lengths':list(range(2,9)),'delays_ms':DELAYS.tolist(),'scales_bp':scales.tolist(),'cell_counts':counts.tolist(),'fit_rule':'90th percentile of positive mean log-price error in Q1 only; minimum 100 observations, global training fallback; floor 0.01 bp'}
    (result/'training_scales.json').write_text(json.dumps(model,indent=2));return scales

def evaluate_prices(result):
    records=[d for d in entries(result) if d.get('status')=='processed' and pd.Timestamp(d['date']).weekday()<5]
    cal=[d for d in records if d['phase']=='calibration' and d['date'] not in EXPLORATORY_DATES];test=[d for d in records if d['phase']=='test']
    rows=[r for alpha in [.1,.05,.01] for r in calibration(result,alpha,False)]
    scales=fit_training_scales(result);normalised_scores={int(delay):[] for delay in DELAYS}
    for d in cal:
        with np.load(result/'days'/(d['date']+'_events.npz')) as e:
            li,ai=scale_index(e['lengths'],e['max_ages'])
            for k,delay in enumerate(DELAYS):
                error=np.maximum(0,(e['gross']-e['archived_realized'][:,k])/e['lengths'])
                normalised_scores[int(delay)].append(float((error/scales[k,li,ai]).max()) if len(error) else 0.)
    for alpha in [.1,.05,.01]:
        for delay in DELAYS:
            rows.append({'delay_ms':int(delay),'method':'scaled_joint_archived','calibration_days':len(cal),'radius_bp':np.nan,'radius_dimensionless':upper_quantile(normalised_scores[int(delay)],alpha),'alpha':alpha,'unknown_day_count':0})
    pd.DataFrame(rows).to_csv(result/'calibration.csv',index=False)
    tests=[];daily=[];fee=-10000*np.log1p(-.1/10000)
    for row in rows:
        delay=int(row['delay_ms']);k=list(DELAYS).index(delay);method=row['method'];q=row['radius_dimensionless'] if method=='scaled_joint_archived' else row['radius_bp']
        npoint=coverpoint=accepted=bad=missing=0;coverdays=0;effective_sum=0.;realized_sum=0.
        for d in test:
            with np.load(result/'days'/(d['date']+'_events.npz')) as e:
                li,ai=scale_index(e['lengths'],e['max_ages']);radius=q*scales[k,li,ai] if method=='scaled_joint_archived' else np.full(len(li),q)
                target=e['realized'][:,k] if method in ['joint_block','edge_block','pooled_points'] else e['archived_realized'][:,k]
                error=np.full(len(li),np.inf);ok=np.isfinite(target);error[ok]=np.maximum(0,(e['gross'][ok]-target[ok])/e['lengths'][ok])
                coverage=(error<=radius);accept=(e['gross']/e['lengths']-fee-radius)>1e-7
                losses=accept&((e['archived_realized'][:,k]-e['lengths']*fee)<=0)
                unknown=accept&~np.isfinite(e['realized'][:,k])
                npoint+=len(li);coverpoint+=int(coverage.sum());accepted+=int(accept.sum());bad+=int(losses.sum());missing+=int(unknown.sum())
                coverdays+=int(coverage.all());effective_sum+=float(radius.sum()) if np.isfinite(q) else 0.
                net=e['archived_realized'][accept,k]-e['lengths'][accept]*fee;realized_sum+=float((10000*np.expm1(net/10000)).sum())
                daily.append({'date':d['date'],'delay_ms':delay,'method':method,'alpha':row['alpha'],'candidate_frames':len(li),'accepted_frames':int(accept.sum()),'losing_quote_frames':int(losses.sum()),'strict_future_unavailable':int(unknown.sum()),'block_covered':int(coverage.all()) if np.isfinite(q) else None,'vacuous':int(not np.isfinite(q))})
        tests.append({**row,'test_days':len(test),'test_frames':npoint,'point_coverage':coverpoint/npoint if npoint and np.isfinite(q) else np.nan,'block_coverage':coverdays/len(test) if test and np.isfinite(q) else np.nan,
                      'mean_radius_bp':effective_sum/npoint if npoint and np.isfinite(q) else np.inf,'accepted_frames':accepted,'losing_quote_frames':bad,'strict_future_unavailable':missing,'loss_fraction_given_accepted':bad/accepted if accepted else np.nan,'mean_archived_return_if_accepted_bp':realized_sum/accepted if accepted else np.nan,'vacuous':int(not np.isfinite(q))})
    pd.DataFrame(tests).to_csv(result/'price_test.csv',index=False);pd.DataFrame(daily).to_csv(result/'price_test_daily.csv',index=False)

def cash_summary(result):
    frame=pd.read_csv(result/'daily_policies.csv');test=frame[(frame.phase=='test')&(frame.weekday<5)]
    rows=[]
    for method,g in test.groupby('method'):
        accepted=int(g.accepted.sum());closed=accepted-int(g.unclosed.sum())
        rows.append({'method':method,'test_days':len(g),'accepted':accepted,'complete':int(g.complete.sum()),'unclosed':int(g.unclosed.sum()),'losses':int(g.losses.sum()),'known_pnl_usd':g.known_pnl_usd.sum(),'conservative_pnl_usd':g.conservative_pnl_usd.sum(),'traded_days':int(g.accepted_day.sum()),'losing_or_unknown_days':int(g.any_loss.sum()),'loss_fraction_among_closed':g.losses.sum()/closed if closed else np.nan,'notional_usd':g.notional_usd.iloc[0]})
    validation=json.loads((result/'policy_validation.json').read_text())
    if validation['selected_policy'] is None:
        rows.append({'method':'validated_policy','test_days':test.date.nunique(),'accepted':0,'complete':0,'unclosed':0,'losses':0,'known_pnl_usd':0,'conservative_pnl_usd':0,'traded_days':0,'losing_or_unknown_days':0,'loss_fraction_among_closed':np.nan,'notional_usd':100000})
    else:
        selected=next(r.copy() for r in rows if r['method']==validation['selected_policy']);selected['method']='validated_policy';rows.append(selected)
    pd.DataFrame(rows).to_csv(result/'cash_test.csv',index=False)

def run():
    for name in ['histdata_year','dukascopy']:
        result=Path('results')/name
        summary=dataset_summary(result);print(name,summary,flush=True)
        evaluate_prices(result)
        if name=='dukascopy':cash_summary(result)

if __name__=='__main__':run()
