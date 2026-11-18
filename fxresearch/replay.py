"""Historical top-of-book replay with cash, depth consumption and no overlap.

Every fill produced here is HYPOTHETICAL. Unknown depth prevents assessment
unless an explicit assumed size is supplied. Actual execution is checked only
by audit-fills on external order logs.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import csv
import json
import math
import sqlite3
from .data import iso_utc


def replay_alerts(database, alerts_path, config, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(database)
    funding=config.get('funding_currency','USD')
    notional=float(config.get('notional',100000))
    capital=float(config.get('capital_per_venue',1000000))
    fee_bp=float(config.get('fee_bp_per_conversion',.02))
    delay=int(config.get('latency_ms_per_leg',100)*1e6)
    cleanup_delay=int(config.get('cleanup_latency_ms',100)*1e6)
    age_limit=int(config.get('execution_max_age_ms',1000)*1e6)
    assumed_size=config.get('assumed_size_base')
    order_type=config.get('order_type','FOK').upper()
    if order_type not in ('IOC','FOK') or notional<=0 or capital<notional or delay<0:
        raise ValueError('invalid replay cash or order parameters')
    accounts=defaultdict(lambda:capital)
    busy=defaultdict(lambda:-1)
    used=defaultdict(float)
    counts=Counter();attempt_count=order_count=0
    latest_consumption={}
    attempt_stream=(output/'replay_attempts.csv').open('w',newline='')
    attempt_fields=['decision_id','timestamp_utc','venue','status','realized_pnl','funding_currency','initial_notional','assumed_execution','terminal_inventory_json']
    attempt_writer=csv.DictWriter(attempt_stream,fieldnames=attempt_fields);attempt_writer.writeheader()
    order_stream=(output/'hypothetical_order_log.csv').open('w',newline='')
    order_fields=['episode_id','initial_amount','leg','timestamp_ns','source_currency','destination_currency','source_amount','destination_amount','fee_amount','fee_currency','quote_id','status','assumed_execution','size_assumed','firm_known_true','receive_time_assumed']
    order_writer=csv.DictWriter(order_stream,fieldnames=order_fields);order_writer.writeheader()
    def push(record):
        nonlocal attempt_count
        attempt_writer.writerow(record);counts[record['status']]+=1;attempt_count+=1
    declared_pairs=set(conn.execute('SELECT DISTINCT venue,base,quote FROM quotes'))

    def quote(i,j,venue,t):
        if (venue,i,j) in declared_pairs:
            base,other,side=i,j,'bid'
        elif (venue,j,i) in declared_pairs:
            base,other,side=j,i,'ask'
        else:return None
        row=conn.execute('SELECT market_ns,bid,ask,bid_size,ask_size,quote_id,firm,receive_assumed FROM quotes WHERE venue=? AND base=? AND quote=? AND market_ns<=? ORDER BY market_ns DESC,id DESC LIMIT 1',(venue,base,other,int(t))).fetchone()
        if row is None or not 0<=t-row[0]<=age_limit:return None
        stamp,bid,ask,bsize,asize,qid,firm,recv_assumed=row
        raw_size=bsize if side=='bid' else asize
        size_assumed=raw_size is None and assumed_size is not None
        raw_size=float(assumed_size) if size_assumed else raw_size
        rate=bid if side=='bid' else 1/ask
        cap=None if raw_size is None else raw_size*(1 if side=='bid' else ask)
        consumption_key=(venue,base,other,qid,stamp,side)
        instrument_side=(venue,base,other,side)
        previous=latest_consumption.get(instrument_side)
        if previous is not None and previous!=consumption_key:used.pop(previous,None)
        latest_consumption[instrument_side]=consumption_key
        if cap is not None:cap=max(0,cap-used[consumption_key])
        return {'rate':rate,'capacity':cap,'qid':qid,'side':side,'size_assumed':size_assumed,
                'firm':firm,'receive_assumed':recv_assumed,'market_ns':stamp,'consumption_key':consumption_key}

    def fill(ep,leg,i,j,venue,t,amount,wallet,cleanup=False):
        nonlocal order_count
        q=quote(i,j,venue,t)
        if q is None:return 'no_fresh_quote',0.
        if q['capacity'] is None:return 'unknown_size',0.
        if order_type=='FOK' and q['capacity']+1e-8<amount and not cleanup:
            amount_filled=0.
        else:amount_filled=min(amount,q['capacity'])
        gross=amount_filled*q['rate'];net=gross*math.exp(-fee_bp/10000)
        wallet[i]-=amount_filled;wallet[j]+=net
        used[q['consumption_key']]+=amount_filled
        status='filled' if amount_filled>=amount-1e-8 else 'partial' if amount_filled>0 else 'rejected'
        order_writer.writerow({'episode_id':ep,'initial_amount':notional,'leg':leg,'timestamp_ns':t,
                       'source_currency':i,'destination_currency':j,'source_amount':amount_filled,
                       'destination_amount':gross,'fee_amount':gross-net,'fee_currency':j,
                       'quote_id':q['qid'],'status':status,'assumed_execution':1,
                       'size_assumed':int(q['size_assumed']),'firm_known_true':int(q['firm']==1),
                       'receive_time_assumed':q['receive_assumed']})
        order_count+=1
        return status,amount_filled

    with open(alerts_path,newline='') as stream:
        for alert in csv.DictReader(stream):
            timestamp=int(alert['timestamp_ns']);cycle=json.loads(alert['cycle_json'])
            ep=str(alert['decision_id']);venues={x[0] for x in cycle}
            record={'decision_id':ep,'timestamp_utc':iso_utc(timestamp),'venue':'','status':'',
                    'realized_pnl':None,'funding_currency':funding,'initial_notional':notional,
                    'assumed_execution':1,'terminal_inventory_json':'{}'}
            if len(venues)!=1:
                record['status']='unsupported_cross_venue_transfer';push(record);continue
            venue=next(iter(venues));record['venue']=venue
            symbols=[x[1] for x in cycle]
            if funding not in symbols:
                record['status']='no_funded_currency_on_cycle';push(record);continue
            k=symbols.index(funding);symbols=symbols[k:]+symbols[:k]
            if timestamp<busy[venue]:
                record['status']='overlap_skipped';push(record);continue
            if accounts[venue]<notional:
                record['status']='insufficient_capital';push(record);continue
            # Do not begin if the initially observed source sizes are unknown.
            pre=[quote(i,j,venue,timestamp) for i,j in zip(symbols,symbols[1:]+symbols[:1])]
            if any(q is None or q['capacity'] is None for q in pre):
                record['status']='unassessable_initial_quote_or_size';push(record);continue
            accounts[venue]-=notional
            wallet=defaultdict(float);wallet[funding]=notional
            current=timestamp;complete=True
            for leg,(i,j) in enumerate(zip(symbols,symbols[1:]+symbols[:1])):
                current+=delay
                status,_=fill(ep,leg,i,j,venue,current,wallet[i],wallet)
                if status!='filled':complete=False;break
            if not complete:
                for symbol in list(wallet):
                    if symbol!=funding and wallet[symbol]>1e-8:
                        current+=cleanup_delay
                        fill(ep,'cleanup',symbol,funding,venue,current,wallet[symbol],wallet,True)
            residual={c:x for c,x in wallet.items() if c!=funding and abs(x)>1e-8}
            busy[venue]=current
            accounts[venue]+=wallet[funding]
            if residual:
                # Freeze this venue account: do not ignore unresolved exposure.
                busy[venue]=2**63-1
                record['status']='unclosed_inventory_account_frozen'
            else:
                duration_seconds=(current-timestamp)/1e9
                financing=notional*float(config.get('funding_rate_per_year',0))*duration_seconds/(365.25*86400)
                accounts[venue]-=financing
                record['realized_pnl']=wallet[funding]-notional-financing
                record['status']='closed_full' if complete else 'closed_after_failure'
            record['terminal_inventory_json']=json.dumps(dict(wallet))
            push(record)
    conn.close();attempt_stream.close();order_stream.close()
    manifest={'config':config,'attempts_or_skips':attempt_count,'orders':order_count,
              'terminal_cash_by_venue':dict(accounts),'status_counts':dict(counts),
              'interpretation':'all fills hypothetical; this is not evidence of actual order execution',
              'depth_model':'top of book, capacity consumed per quote id and side until a new quote id',
              'limitations':['no endogenous market impact beyond displayed top depth','no venue-specific last-look model',
                             'raw trades, queue position and cancellations are not reconstructed from bid/ask alone']}
    (output/'replay_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
