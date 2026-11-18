"""Reproduce experiments 4--6 and scalability checks; synthetic data are labelled.

Run from the project directory: python experiments_extension.py
Every table/figure in the appended report is generated from these outputs.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import time
import numpy as np
import pandas as pd
from fxresearch.certification import (structured_radius, box_radius, conformal_quantile,
    clopper_pearson, adaptive_coverage, minimum_blocks, conditional_risk_upper,
    ReservedLeg,reserved_path_bound)
from fxresearch.execution import MarketTape, ExecutionConfig, execute_route, price_only_return
from fxresearch.flow import CashEdge, funded_flow
from fxresearch.graph import enumerate_cycles, positive_cycle, maximum_cycle_mean, cycle_mean


def csv_write(path, rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def experiment_four(output, seed=131401, cases=2000):
    rng=np.random.default_rng(seed)
    # Path 0->1->2->3, reference 0->3, and two additional paths/chords.
    # Cycle-space rank is three; projected-box is a stronger comparator.
    incidence=np.array([[-1,0,0,-1,-1,0],[1,-1,0,0,0,-1],
                        [0,1,-1,0,1,0],[0,0,1,1,0,1.]])
    c=np.array([1.,1,1,-1,0,0])
    projector=np.eye(6)-incidence.T@np.linalg.pinv(incidence@incidence.T)@incidence
    assert np.max(np.abs(incidence@c))==0
    rows=[];maximum_bound_violation=0.;gauge_error=0.
    for case in range(cases):
        common=rng.normal(0,.5,(4,2))
        factors=incidence.T@common+rng.normal(0,.025,(6,2))
        remainder=rng.uniform(.015,.04,6)
        radius=structured_radius(c,factors,remainder)
        box=box_radius(c,factors,remainder)
        projected=float(np.abs(c)@(np.linalg.norm(projector@factors,axis=1)+np.abs(projector)@remainder))
        assert projected+1e-10>=radius
        nominal=float(rng.uniform(-.1,.85))
        capacity_good=rng.random()>.15
        # Actual source-unit capacity floors, checked against upper balances.
        edge_upper=np.linalg.norm(factors[:3],axis=1)+remainder[:3]
        edge_lower=-edge_upper
        upper_balances=100000*np.exp(np.r_[0,np.cumsum(nominal/3+edge_upper)[:2]]/10000)
        floors=upper_balances*1.05
        if not capacity_good:floors[1]=.8*upper_balances[1]
        legs=[ReservedLeg(math.exp((nominal/3+edge_lower[k])/10000),
                          math.exp((nominal/3+edge_upper[k])/10000),floors[k]) for k in range(3)]
        checked=reserved_path_bound(100000,legs)
        assert checked['capacity_valid']==capacity_good
        z=rng.normal(size=(128,2));z/=np.maximum(1,np.linalg.norm(z,axis=1))[:,None]
        eps=rng.uniform(-1,1,(128,6))*remainder
        errors=z@factors.T+eps
        realized=nominal+errors@c
        lower=nominal-radius
        maximum_bound_violation=max(maximum_bound_violation,float(lower-realized.min()))
        phi=rng.normal(0,50,4)
        gauge_error=max(gauge_error,float(np.max(np.abs((errors+incidence.T@phi)@c-errors@c))))
        rows.append({'case':case,'nominal_gain_bp':nominal,'factor_radius_bp':radius,'box_radius_bp':box,
                     'projected_box_radius_bp':projected,'projected_box_accept':int(nominal-projected>0 and capacity_good),
                     'capacity_valid':int(capacity_good),'factor_accept':int(lower>0 and capacity_good),
                     'box_accept':int(nominal-box>0 and capacity_good),
                     'lower_gain_bp':lower,'minimum_sample_gain_bp':realized.min(),
                     'sample_violation':int(lower>realized.min()+1e-10),
                     'minimum_capacity_ratio':float(np.min(floors/upper_balances))})
    csv_write(output/'exp4_bounds.csv',rows)
    flow_rows=[]
    for case in range(100):
        amount=100000.
        rates=np.exp(rng.uniform(-.4,.7,3)/10000)
        edges=[CashEdge(0,1,rates[0],rng.uniform(.3,1.2)*amount),
               CashEdge(1,2,rates[1],rng.uniform(.3,1.2)*amount),
               CashEdge(2,3,rates[2],rng.uniform(.3,1.2)*amount),
               CashEdge(0,3,1,amount)]
        base=funded_flow(4,edges,0,3,amount)
        units=np.exp(rng.uniform(-2,2,4))
        transformed=[CashEdge(e.source,e.destination,e.rate*units[e.destination]/units[e.source],e.capacity*units[e.source]) for e in edges]
        other=funded_flow(4,transformed,0,3,amount*units[0],units[3]/units[0])
        flow_rows.append({'case':case,'profit_usd':base['net_profit'],'dual_gap_usd':base['duality_gap'],
                          'unit_error_usd':abs(other['net_profit']/units[3]-base['net_profit'])})
    csv_write(output/'exp4_flow.csv',flow_rows)
    result={'seed':seed,'cases':cases,'shocks_per_case':128,
            'factor_accepts':sum(r['factor_accept'] for r in rows),
            'box_accepts':sum(r['box_accept'] for r in rows),
            'projected_box_accepts':sum(r['projected_box_accept'] for r in rows),
            'capacity_rejections':sum(not r['capacity_valid'] for r in rows),
            'median_width_ratio':float(np.median([r['box_radius_bp']/r['factor_radius_bp'] for r in rows])),
            'median_projected_ratio':float(np.median([r['projected_box_radius_bp']/r['factor_radius_bp'] for r in rows])),
            'maximum_bound_violation_bp':maximum_bound_violation,'gauge_error_bp':gauge_error,
            'maximum_duality_gap_usd':max(abs(r['dual_gap_usd']) for r in flow_rows),
            'maximum_flow_unit_error_usd':max(r['unit_error_usd'] for r in flow_rows),
            'status':'controlled known uncertainty set; no learned market guarantee'}
    (output/'exp4_summary.json').write_text(json.dumps(result,indent=2))
    print('Experiment 4',result,flush=True)


def error_blocks(rng, blocks, times=32, actions=24, volatility=1):
    scale=np.linspace(.15,.65,actions)
    factor=np.zeros((blocks,times))
    factor[:,0]=rng.normal(size=blocks)
    for t in range(1,times):factor[:,t]=.85*factor[:,t-1]+math.sqrt(1-.85**2)*rng.normal(size=blocks)
    signs=np.r_[np.ones(actions//2),-np.ones(actions-actions//2)]
    errors=volatility*scale[None,None,:]*(.8*factor[:,:,None]*signs+.6*rng.normal(size=(blocks,times,actions)))
    truth=np.full_like(errors,-.20)
    for b in range(blocks):
        if rng.random()<.5:
            candidate=int(rng.integers(actions))
            truth[b,:,candidate]=2.0
    return errors,truth,scale


def experiment_five(output,seed=131501,calibration=199,test_blocks=400):
    rng=np.random.default_rng(seed)
    train,_,_=error_blocks(rng,80)
    cal,_,_=error_blocks(rng,calibration)
    # Scale is trained independently, never fit on calibration/test labels.
    scales=np.maximum(.01,train.std(axis=(0,1)))
    alpha=.05
    marginal=float(np.quantile(cal,1-alpha,method='higher'))
    per_action=np.array([conformal_quantile(cal[:,:,i].max(axis=1),alpha) for i in range(cal.shape[-1])])
    joint=conformal_quantile(cal.max(axis=(1,2)),alpha)
    standardized=conformal_quantile((cal/scales).max(axis=(1,2)),alpha)*scales
    radii={'marginal_observation':np.full(cal.shape[-1],marginal),
           'fixed_action_block':per_action,'joint_block':np.full(cal.shape[-1],joint),
           'scaled_joint_block':standardized}
    rows=[];blocks_rows=[]
    for regime,volatility in [('exchangeable',1.),('volatility_shift',2.5)]:
        errors,truth,_=error_blocks(rng,test_blocks,volatility=volatility)
        nominal=truth+errors
        for name,radius in radii.items():
            lower=nominal-radius
            chosen=lower.argmax(axis=2)
            accepted=lower.max(axis=2)>0
            selected_truth=np.take_along_axis(truth,chosen[:,:,None],axis=2)[:,:,0]
            bad=accepted&(selected_truth<=0)
            covered=np.all(errors<=radius,axis=(1,2))
            bad_blocks=bad.any(axis=1)
            lo,hi=clopper_pearson(int(bad_blocks.sum()),test_blocks)
            rows.append({'regime':regime,'method':name,'blocks':test_blocks,'alpha':alpha,
                         'mean_radius_bp':float(radius.mean()),'block_coverage':float(covered.mean()),
                         'bad_block_rate':float(bad_blocks.mean()),'bad_block_ci_low':lo,'bad_block_ci_high':hi,
                         'accepted_attempts':int(accepted.sum()),'bad_accepts':int(bad.sum()),
                         'conditional_bad_fraction':float(bad.sum()/accepted.sum()) if accepted.any() else None,
                         'mean_gain_per_opportunity_bp':float(np.where(accepted,selected_truth,0).mean())})
            for b in range(test_blocks):
                blocks_rows.append({'regime':regime,'method':name,'block':b,'covered':int(covered[b]),
                                    'bad_block':int(bad_blocks[b]),'accepts':int(accepted[b].sum()),'bad_accepts':int(bad[b].sum())})
    csv_write(output/'exp5_methods.csv',rows);csv_write(output/'exp5_blocks.csv',blocks_rows)
    # Online stress: labels are observed only after each complete block.
    before,_,_=error_blocks(rng,1500)
    after,_,_=error_blocks(rng,1500,volatility=2.5)
    online_scores=np.r_[before.max(axis=(1,2)),after.max(axis=(1,2))]
    initial=cal.max(axis=(1,2))
    adaptive=adaptive_coverage(online_scores,initial,alpha,gamma=.005,window=199)
    fixed_errors=online_scores>joint
    online_rows=[]
    for t,(score,row) in enumerate(zip(online_scores,adaptive)):
        online_rows.append({'block':t,'score':score,'fixed_radius':joint,'adaptive_radius':row[0],
                            'fixed_error':int(fixed_errors[t]),'adaptive_error':int(row[1]),
                            'beta_before':row[2],'beta_after':row[3]})
    csv_write(output/'exp5_online.csv',online_rows)
    identity_error=abs(adaptive[:,1].mean()-alpha-(alpha-adaptive[-1,3])/(.005*len(adaptive)))
    u=rng.random(100000);selected=u>.95
    sample_rows=[{'alpha':a,'blocks_available':m,'minimum_blocks':minimum_blocks(a),
                  'infinite_radius':int(math.isinf(conformal_quantile(np.arange(m,dtype=float),a)))}
                 for a in [.2,.1,.05,.01,.0002] for m in [5,19,99,199]]
    csv_write(output/'exp5_sample_requirements.csv',sample_rows)
    summary={'seed':seed,'training_blocks':80,'calibration_blocks':calibration,'test_blocks_per_regime':test_blocks,
             'times_per_block':32,'actions':24,'alpha':alpha,'joint_radius_bp':joint,
             'online_blocks':len(adaptive),'online_fixed_error_frequency':float(fixed_errors.mean()),
             'online_adaptive_error_frequency':float(adaptive[:,1].mean()),'aci_identity_error':identity_error,
             'aci_infinite_intervals':int(np.isposinf(adaptive[:,0]).sum()),
             'aci_empty_intervals':int(np.isneginf(adaptive[:,0]).sum()),
             'selection_counterexample_marginal_coverage':float((~selected).mean()),
             'selection_counterexample_bad_among_selected':1.0,
             'existing_five_calibration_windows_radius_at_one_percent':'infinite',
             'status':'synthetic prediction-error panels; exchangeability is imposed only in the first regime'}
    (output/'exp5_summary.json').write_text(json.dumps(summary,indent=2))
    print('Experiment 5',summary,flush=True)


def make_episode(seed,n=6):
    rng=np.random.default_rng(seed)
    step=10;end=7000;decision=2000
    sigma=np.r_[0.,np.linspace(.8,1.4,n-1)]
    increments=rng.normal(size=(end//step+1,n))*sigma*math.sqrt(step/1000)
    increments-=.5*sigma**2/10000*(step/1000)
    increments[:,0]=0
    initial=np.log([1,1.09,1.27,1/150,1.12,.66][:n])*10000
    values=initial+np.cumsum(increments,axis=0)
    scenario=int(rng.integers(3))
    amplitude=[0,6.,6.][scenario]
    duration=[0,40,2000][scenario]
    tape=MarketTape(values,pulse_start_ms=np.array([decision-1200]),
                    pulse_end_ms=np.array([decision+duration]),pulse_amplitudes_bp=np.array([amplitude]))
    observed=np.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i+1,n):
            age=min(1500,int(rng.exponential(100))+(500 if (i,j)==(0,1) else 0))
            observed[i,j]=observed[j,i]=decision-age
    return tape,decision,observed,['null','short_pulse','persistent_pulse'][scenario]


def nominal_routes(tape,routes,observed,fee=.02):
    gains=[];ages=[]
    for route in routes:
        gain=0.;age=0
        for i,j in zip(route,route[1:]+route[:1]):
            stamp=int(observed[i,j]);gain+=tape.mid_bp(i,j,stamp)-tape.half_spread_bp-fee
            age=max(age,2000-stamp)
        gains.append(10000*math.expm1(gain/10000));ages.append(age)
    return np.asarray(gains),np.asarray(ages)


def experiment_six(output,seed=131601,calibration=199,test_episodes=600):
    routes=list(enumerate_cycles(6,max_length=3,anchor=0))
    configs={mode:ExecutionConfig(mode=mode) for mode in ['reserved','unreserved']}
    calibration_rows=[];thresholds={};policy_rows=[];chosen_policies={}
    policy_thresholds=(0.,.5,1.,2.,3.,4.,6.,8.)  # Frozen before labels.
    for mode,config in configs.items():
        price_scores=[];execution_scores=[];proposals=[]
        for episode in range(calibration):
            tape,decision,observed,scenario=make_episode(seed+episode)
            nominal,_=nominal_routes(tape,routes,observed)
            price=np.array([price_only_return(tape,r,decision,config,seed*100000+episode*100+index) for index,r in enumerate(routes)])
            actual=np.array([execute_route(tape,r,decision,observed,config,seed*100000+episode*100+index)['return_bp'] for index,r in enumerate(routes)])
            sp=max(0,float((nominal-price).max()));se=max(0,float((nominal-actual).max()))
            price_scores.append(sp);execution_scores.append(se)
            calibration_rows.append({'mode':mode,'episode':episode,'scenario':scenario,'price_score_bp':sp,'execution_score_bp':se})
            chosen=int(nominal.argmax())
            proposals.append((float(nominal[chosen]),float(actual[chosen])))
        thresholds[mode]={'price':conformal_quantile(price_scores,.05),'execution':conformal_quantile(execution_scores,.05)}
        feasible=[]
        for threshold in policy_thresholds:
            included=[y for pred,y in proposals if pred>threshold]
            losses=sum(y < -1e-8 for y in included)
            upper=conditional_risk_upper(losses,len(included),delta=.05,family_size=len(policy_thresholds))
            objective=sum(included)/calibration
            valid=upper<=.05
            policy_rows.append({'mode':mode,'threshold_bp':threshold,'calibration_accepted':len(included),
                                'calibration_losses':losses,'conditional_risk_upper':upper,
                                'valid_at_alpha_05':int(valid),'calibration_gain_per_opportunity_bp':objective})
            if valid:feasible.append((objective,threshold,upper))
        winner=max(feasible) if feasible else None
        chosen_policies[mode]={'threshold_bp':winner[1] if winner else None,
                               'risk_upper':winner[2] if winner else None,
                               'decision':'validated' if winner else 'abstain_no_validated_policy'}
    csv_write(output/'exp6_calibration.csv',calibration_rows)
    csv_write(output/'exp6_conditional_policies.csv',policy_rows)
    rows=[];traces=[]
    for episode in range(test_episodes):
        tape,decision,observed,scenario=make_episode(seed+100000+episode)
        nominal,ages=nominal_routes(tape,routes,observed)
        for mode,config in configs.items():
            for method in ['nominal','age_100ms','price_certificate','execution_certificate','conditional_risk_policy']:
                lower=nominal.copy()
                if method=='age_100ms':lower[ages>100]=-np.inf
                if method=='price_certificate':lower-=thresholds[mode]['price']
                if method=='execution_certificate':lower-=thresholds[mode]['execution']
                if method=='conditional_risk_policy':
                    cutoff=chosen_policies[mode]['threshold_bp']
                    lower-=cutoff if cutoff is not None else np.inf
                index=int(lower.argmax());accept=bool(lower[index]>0)
                if accept:
                    result=execute_route(tape,routes[index],decision,observed,config,
                                         seed*100000+(100000+episode)*100+index,keep_trace=episode<10)
                    pnl=result['pnl'];ret=result['return_bp'];complete=result['complete']
                else:
                    result={};pnl=ret=0.;complete=None
                rows.append({'mode':mode,'method':method,'episode':episode,'scenario':scenario,
                             'accepted':int(accept),'route':str(routes[index]) if accept else '',
                             'nominal_return_bp':float(nominal[index]) if accept else 0,
                             'lower_return_bp':float(lower[index]) if np.isfinite(lower[index]) else None,
                             'pnl_usd':pnl,'return_bp':ret,'loss':int(ret < -1e-8),
                             'complete':int(complete) if complete is not None else None})
                if accept and episode<10:
                    for event in result['trace']:
                        traces.append({'mode':mode,'method':method,'episode':episode,**event})
    csv_write(output/'exp6_attempts.csv',rows)
    if traces:csv_write(output/'exp6_order_trace.csv',traces)
    frame=pd.DataFrame(rows);summary=[]
    for (mode,method),group in frame.groupby(['mode','method'],sort=False):
        accepted=group['accepted'].sum();losses=group['loss'].sum()
        by_scenario={k:float(v['return_bp'].mean()) for k,v in group.groupby('scenario')}
        lo,hi=clopper_pearson(int(losses),len(group))
        se=float(group['return_bp'].std(ddof=1)/math.sqrt(len(group)))
        summary.append({'mode':mode,'method':method,'test_episodes':len(group),'accepted':int(accepted),'losses':int(losses),
                        'attempt_loss_probability':float(losses/len(group)),'loss_prob_ci_low':lo,'loss_prob_ci_high':hi,
                        'loss_fraction_if_accepted':float(losses/accepted) if accepted else None,
                        'mean_return_per_opportunity_bp':float(group['return_bp'].mean()),
                        'mean_ci_low_bp':float(group['return_bp'].mean())-1.96*se,
                        'mean_ci_high_bp':float(group['return_bp'].mean())+1.96*se,
                        'net_pnl_usd':float(group['pnl_usd'].sum()),
                        'complete_fraction_if_accepted':float(group.loc[group.accepted==1,'complete'].mean()) if accepted else None,
                        'null_mean_bp':by_scenario.get('null'),'short_pulse_mean_bp':by_scenario.get('short_pulse'),
                        'persistent_pulse_mean_bp':by_scenario.get('persistent_pulse')})
    csv_write(output/'exp6_methods.csv',summary)
    meta={'seed':seed,'currencies':6,'routes':len(routes),'max_cycle_length':3,'funding_currency':'USD',
          'initial_notional_usd':100000,'calibration_episodes':calibration,'test_episodes':test_episodes,'alpha':.05,
          'thresholds_bp':thresholds,'conditional_policies':chosen_policies,
          'conditional_risk_alpha':.05,'calibration_confidence_delta':.05,
          'frozen_policy_thresholds_bp':policy_thresholds,
          'configurations':{k:asdict(v) for k,v in configs.items()},
          'scenarios':{'null':{'amplitude_bp':0},'short_pulse':{'amplitude_bp':6,'duration_after_decision_ms':40},
                       'persistent_pulse':{'amplitude_bp':6,'duration_after_decision_ms':2000}},
          'assumptions':['independent episodes; one attempt at most in each episode',
                         'common random numbers for the same potential route across methods',
                         'cleanup liquidity is replenished at the last depth level',
                         'exogenous prices; no endogenous response of other traders',
                         'reserved mode assumes a full-route commitment, not merely IOC/FOK on each leg'],
          'not_claimed':'real-market profits or a validated model of a named venue'}
    (output/'exp6_protocol.json').write_text(json.dumps(meta,indent=2))
    print('Experiment 6',json.dumps(summary),flush=True)


def scalability(output,seed=131701):
    rng=np.random.default_rng(seed);rows=[]
    for n in [4,8,16,32,64]:
        for density in [1.,.2]:
            h=rng.normal(0,100,n);a=h[None,:]-h[:,None]-.1
            a[rng.random((n,n))>density]=-np.inf;np.fill_diagonal(a,-np.inf)
            # Keep one ring to ensure a nontrivial SCC for sparse cases.
            for i in range(n):a[i,(i+1)%n]=h[(i+1)%n]-h[i]-.1
            bf=[];karp=[]
            for _ in range(5):
                began=time.perf_counter();w=positive_cycle(a);bf.append(time.perf_counter()-began)
                began=time.perf_counter();mu=maximum_cycle_mean(a);karp.append(time.perf_counter()-began)
                assert not w and abs(mu+.1)<1e-9
            rows.append({'currencies':n,'edges':int(np.isfinite(a).sum()),'density_target':density,
                         'bf_median_ms':1000*np.median(bf),'karp_median_ms':1000*np.median(karp),
                         'dp_storage_bytes':8*(n+1)*n,'mean_bp':mu})
    csv_write(output/'scalability.csv',rows)
    print('Scalability complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/extension'))
    parser.add_argument('--experiments',nargs='+',default=['4','5','6','scale'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    if '4' in args.experiments:experiment_four(args.output)
    if '5' in args.experiments:experiment_five(args.output)
    if '6' in args.experiments:experiment_six(args.output)
    if 'scale' in args.experiments:scalability(args.output)
    manifest={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'elapsed_seconds':time.perf_counter()-began,'experiments':args.experiments,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_provenance':'experiments 4,5,6 are synthetic; legacy market results are not modified'}
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
