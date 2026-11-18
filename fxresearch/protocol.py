"""Frozen chronological plans and hypothetical policy-block evaluation."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import math
import sqlite3
from .data import iso_utc,NS
from .replay import replay_alerts

DAY_NS=86400*NS


def observed_days(database):
    conn=sqlite3.connect(database)
    days=[int(row[0]) for row in conn.execute('SELECT DISTINCT receive_ns / ? AS utc_day FROM quotes ORDER BY utc_day',(DAY_NS,))]
    conn.close()
    return days


def split_plan(database,output,training_days=60,calibration_days=99,test_days=30,step_days=30,purge_ms=5000):
    days=observed_days(database)
    if min(training_days,calibration_days,test_days,step_days)<1 or purge_ms<0:
        raise ValueError('positive split lengths and nonnegative purge required')
    windows=[]
    total=training_days+calibration_days+test_days
    for offset in range(0,len(days)-total+1,step_days):
        slices=[days[offset:offset+training_days],
                days[offset+training_days:offset+training_days+calibration_days],
                days[offset+training_days+calibration_days:offset+total]]
        windows.append({'fold':len(windows),'training_dates':[iso_utc(d*DAY_NS)[:10] for d in slices[0]],
                        'calibration_dates':[iso_utc(d*DAY_NS)[:10] for d in slices[1]],
                        'test_dates':[iso_utc(d*DAY_NS)[:10] for d in slices[2]],
                        'label_rule':'every execution/cleanup endpoint must stay inside the decision block',
                        'last_decision_each_day_offset_ns':DAY_NS-int(purge_ms*1e6)})
    plan={'observed_utc_dates':len(days),'training_days':training_days,'calibration_days':calibration_days,
          'test_days':test_days,'step_days':step_days,'purge_ms':purge_ms,'windows':windows,
          'test_windows_overlap':step_days<test_days,
          'warning':'dates with observations are not automatically full or independent market days',
          'status':'ready' if windows else 'insufficient_days_for_requested_plan'}
    Path(output).write_text(json.dumps(plan,indent=2));return plan


def evaluate_hypothetical_policies(database,alerts,registry_path,replay_config,output):
    """Evaluate each precommitted policy independently on each observed UTC day.

    The target is simulated daily any-loss risk. Each day starts with the same
    declared cash; an unresolved position is counted as failure, not a zero.
    These model counterfactuals cannot stand in for unobserved real order fills.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    registry=json.loads(Path(registry_path).read_text())
    day_files={};active_day=None;handle=None;writer=None
    try:
        with open(alerts,newline='') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames
            for row in reader:
                day=int(row['timestamp_ns'])//DAY_NS
                if day!=active_day:
                    if active_day is not None and day<active_day:raise ValueError('alerts must be chronologically sorted')
                    if handle is not None:handle.close()
                    folder=output/'partitions';folder.mkdir(exist_ok=True)
                    path=folder/(iso_utc(day*DAY_NS)[:10]+'.csv');day_files[day]=path
                    handle=path.open('w',newline='');active_day=day
                    writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                writer.writerow(row)
    finally:
        if handle is not None:handle.close()
    rows=[]
    for day in observed_days(database):
        date=iso_utc(day*DAY_NS)[:10]
        for policy in registry['policies']:
            policy_id=policy['policy_id'];folder=output/policy_id/date;folder.mkdir(parents=True,exist_ok=True)
            selected=folder/'selected_alerts.csv'
            with selected.open('w',newline='') as destination:
                writer=csv.DictWriter(destination,fieldnames=fields);writer.writeheader()
                if day in day_files:
                    with day_files[day].open(newline='') as source:
                        for row in csv.DictReader(source):
                            # Convert the log-gain signal into simple return bp.
                            predicted_bp=10000*math.expm1(float(row['gain_bp'])/10000)
                            if predicted_bp<=float(policy['minimum_predicted_return_bp']):continue
                            length=len(json.loads(row['cycle_json']))
                            horizon=int(length*(replay_config.get('latency_ms_per_leg',100)+replay_config.get('cleanup_latency_ms',100))*1e6)
                            if int(row['timestamp_ns'])+horizon>=((day+1)*DAY_NS):continue
                            writer.writerow(row)
            manifest=replay_alerts(database,selected,replay_config,folder)
            attempts=folder/'replay_attempts.csv'
            acted=0;unclosed=0;unknown=loss=False;pnl=0.
            if attempts.exists():
                with attempts.open(newline='') as stream:
                    for record in csv.DictReader(stream):
                        if record['status'] not in ('closed_full','closed_after_failure','unclosed_inventory_account_frozen'):continue
                        acted+=1
                        unknown|=record['status']=='unclosed_inventory_account_frozen'
                        unclosed+=int(record['status']=='unclosed_inventory_account_frozen')
                        if record['realized_pnl']!='':
                            value=float(record['realized_pnl']);pnl+=value;loss|=value<0
            loss|=unknown
            if unknown:pnl-=unclosed*replay_config.get('notional',100000) # Penalise every unresolved episode.
            rows.append({'block_id':date,'policy_id':policy_id,'accepted':int(bool(acted)),
                         'loss':int(loss),'pnl':pnl,'unclosed_conservatively_failed':int(unknown),
                         'target':'hypothetical_replay_daily_any_loss'})
    with (output/'policy_blocks.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['block_id','policy_id','accepted','loss','pnl'])
        writer.writeheader();writer.writerows(rows)
    return {'blocks':len(observed_days(database)),'policies':len(registry['policies']),
            'output':str(output/'policy_blocks.csv'),'target':'hypothetical replay, not real execution',
            'filter_observable':'simple return bp converted from gain_bp, after the detection fee',
            'missing_counterfactual_fills':'modelled only; external order logs are required for a real target'}
