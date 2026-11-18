"""Fetch original public Dukascopy daily tick packets, with an auditable ledger.

No paid service, account, or candle data is used. Failed HTTP requests are
distinct from empty/absent files. No missing packet is filled with invented ticks.
"""
import argparse
import asyncio
from datetime import date,timedelta,datetime,timezone
import hashlib
import json
import lzma
from pathlib import Path
import time
import httpx
import numpy as np

PAIRS='EURUSD EURGBP EURJPY EURCHF EURAUD EURCAD EURNZD GBPUSD GBPJPY GBPCHF GBPAUD GBPCAD GBPNZD USDJPY USDCHF USDCAD AUDUSD AUDJPY AUDCHF AUDCAD AUDNZD CADJPY CADCHF CHFJPY NZDUSD NZDJPY NZDCHF NZDCAD'.split()
CURRENCIES=['USD','EUR','GBP','JPY','CHF','AUD','CAD','NZD']
TICK_DTYPE=np.dtype([('ms','>u4'),('ask','>u4'),('bid','>u4'),('ask_volume','>f4'),('bid_volume','>f4')])

def packet_url(pair,day):
    return f'https://datafeed.dukascopy.com/datafeed/{pair}/{day.year}/{day.month-1:02d}/{day.day:02d}_ticks.bi5'

def decode(payload):
    if not payload:return np.empty(0,dtype=TICK_DTYPE)
    raw=lzma.decompress(payload)
    if len(raw)%20:raise ValueError('packet length not divisible by 20')
    rows=np.frombuffer(raw,dtype=TICK_DTYPE)
    if len(rows) and (rows['ms'].max()>=86400000 or np.any(rows['ms'][1:]<rows['ms'][:-1])):
        raise ValueError('daily timestamp range/order violation')
    return rows


async def collect(args):
    root=args.raw;root.mkdir(parents=True,exist_ok=True)
    ledger=args.ledger;ledger.parent.mkdir(parents=True,exist_ok=True)
    previous={}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:r=json.loads(line);previous[(r['date'],r['pair'])]=r
            except (ValueError,KeyError):pass
    days=[];d=date.fromisoformat(args.start)
    while d<date.fromisoformat(args.end):
        # Saturday is a declared structural FX closure. Sunday is requested.
        if d.weekday()!=5:days.append(d)
        d+=timedelta(days=1)
    queue=asyncio.Queue()
    for d in days:
        for pair in PAIRS:queue.put_nowait((pair,d))
    total=queue.qsize();counts={};started=time.monotonic();finished=0;bytes_total=0
    lock=asyncio.Lock();blocked=asyncio.Event()
    async with httpx.AsyncClient(timeout=httpx.Timeout(75,connect=20),limits=httpx.Limits(max_connections=args.workers,max_keepalive_connections=args.workers),follow_redirects=True) as client:
        with ledger.open('a',buffering=1) as stream:
            async def worker():
                nonlocal finished,bytes_total
                while True:
                    if blocked.is_set():return
                    try:pair,d=queue.get_nowait()
                    except asyncio.QueueEmpty:return
                    key=(str(d),pair);path=root/str(d)/f'{pair}.bi5';old=previous.get(key)
                    reused=False
                    if old and old.get('status')=='ok' and path.exists() and path.stat().st_size==old.get('bytes'):
                        r=old;reused=True
                    elif old and old.get('status') in ('empty','absent') and not args.retry_absent:
                        r=old;reused=True
                    else:
                        r={'date':str(d),'pair':pair,'url':packet_url(pair,d),'retrieved_utc':datetime.now(timezone.utc).isoformat()}
                        for attempt in range(args.retries):
                            try:
                                response=await client.get(r['url'])
                                r.update(http_status=response.status_code,attempts=attempt+1)
                                if response.status_code==429:
                                    r.update(status='rate_limited',provider_message=response.text[:300],retry_after=response.headers.get('Retry-After'))
                                    blocked.set();break
                                if response.status_code==404:
                                    r['status']='absent';break
                                if response.status_code in (429,500,502,503,504):
                                    r['status']='http_error'
                                    delay=float(response.headers.get('Retry-After',min(30,2**attempt)))
                                    await asyncio.sleep(delay);continue
                                response.raise_for_status()
                                payload=response.content;ticks=decode(payload)
                                path.parent.mkdir(parents=True,exist_ok=True)
                                temporary=path.with_suffix('.part');temporary.write_bytes(payload);temporary.replace(path)
                                r.update(status='ok' if len(ticks) else 'empty',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),ticks=len(ticks))
                                if len(ticks):
                                    invalid=(ticks['bid']==0)|(ticks['ask']<ticks['bid'])|~np.isfinite(ticks['ask_volume'])|~np.isfinite(ticks['bid_volume'])|(ticks['ask_volume']<0)|(ticks['bid_volume']<0)
                                    r.update(first_ms=int(ticks['ms'][0]),last_ms=int(ticks['ms'][-1]),invalid_rows=int(invalid.sum()),duplicate_times=int((ticks['ms'][1:]==ticks['ms'][:-1]).sum()),fraction_whole_second=float((ticks['ms']%1000==0).mean()))
                                break
                            except Exception as exc:
                                r.update(status='error',error=type(exc).__name__+': '+str(exc)[:300],attempts=attempt+1)
                                await asyncio.sleep(min(20,2**attempt))
                        else:r['exhausted_retries']=True
                    async with lock:
                        if not reused:stream.write(json.dumps(r)+'\n')
                        counts[r['status']]=counts.get(r['status'],0)+1
                        bytes_total+=r.get('bytes',0);finished+=1
                        if finished%100==0 or finished==total:
                            print(json.dumps({'finished':finished,'total':total,'counts':counts,'MB':round(bytes_total/1e6,1),'seconds':round(time.monotonic()-started,1)}),flush=True)
                    queue.task_done()
            await asyncio.gather(*(worker() for _ in range(args.workers)))
    summary={'start':args.start,'end_exclusive':args.end,'requested_dates':len(days),'pairs':PAIRS,'currencies':CURRENCIES,'counts':counts,'bytes':bytes_total,'seconds':time.monotonic()-started,'packet_time_base':'daily UTC; integer milliseconds','not_collected':'Saturday structural closure; no receiver timestamps or order confirmations'}
    (ledger.parent/'collection_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',default='2024-01-01');p.add_argument('--end',default='2025-01-01')
    p.add_argument('--raw',type=Path,default=Path('raw/dukascopy'))
    p.add_argument('--ledger',type=Path,default=Path('results/dukascopy/source_ledger.jsonl'))
    p.add_argument('--workers',type=int,default=8);p.add_argument('--retries',type=int,default=6)
    p.add_argument('--retry-absent',action='store_true');args=p.parse_args()
    asyncio.run(collect(args))
