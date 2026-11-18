"""Original monthly ASCII tick ZIPs; canonical per-day arrays are derived caches."""
import argparse,csv,hashlib,http.cookiejar,json,re,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,build_opener,HTTPCookieProcessor
from urllib.error import HTTPError
from zipfile import ZipFile
import numpy as np
import pandas as pd
from collect_dukascopy import PAIRS

HOST='https://www.histdata.com'
MONTHS=['202312']+[f'2024{m:02d}' for m in range(1,13)]
STOP=threading.Event();LOCK=threading.Lock()

def download(raw,month,pair,offline=False):
    dest=raw/f'{month}_{pair}.zip';member=f'DAT_ASCII_{pair}_T_{month}.csv'
    url=f'{HOST}/download-free-forex-data/?/ascii/tick-data-quotes/{pair}/{month[:4]}/{int(month[4:])}'
    if not dest.exists():
        if offline:raise FileNotFoundError('Offline reproduction requires original archive: '+str(dest))
        for attempt in range(4):
            if STOP.is_set():raise RuntimeError('provider download limit; stopped')
            try:
                opener=build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
                headers={'User-Agent':'Mozilla/5.0 (compatible; AcademicFXReplay/1.0)'}
                with opener.open(Request(url,headers=headers),timeout=60) as r:html=r.read().decode()
                form=html.split('<form id="file_down"',1)[1].split('</form>',1)[0]
                fields=dict(re.findall(r'name="(\w+)"[^>]*?value="([^"]+)"',form))
                if fields.get('datemonth')!=month or fields.get('fxpair')!=pair:raise ValueError('source form pair/month mismatch')
                request=Request(HOST+'/get.php',data=urlencode(fields).encode(),headers={**headers,'Referer':url,'Origin':HOST,'Content-Type':'application/x-www-form-urlencoded'})
                part=dest.with_suffix('.part')
                with opener.open(request,timeout=120) as r,part.open('wb') as out:
                    while chunk:=r.read(1024*1024):out.write(chunk)
                with ZipFile(part) as z:z.getinfo(member)
                part.replace(dest);break
            except HTTPError as exc:
                if exc.code==429:STOP.set();raise
                if attempt==3:raise
                time.sleep(2**attempt)
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    with ZipFile(dest) as z:z.getinfo(member)
    return {'month':month,'pair':pair,'archive':dest.name,'member':member,'url':url,'bytes':dest.stat().st_size,'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),'status':'ok'}

def extract(source,cache):
    month,pair=source['month'],source['pair'];folder=cache/month/pair;marker=folder/'complete.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        if result['source_sha256']==source['sha256']:return result
        raise ValueError('cached interpretation changed; use a clean cache')
    folder.mkdir(parents=True,exist_ok=True);counts={};raw_count=0
    # An interrupted, uncommitted extraction is rebuilt from the original ZIP.
    for partial in list(folder.glob('*.bin'))+list(folder.glob('*.npz')):partial.unlink()
    with ZipFile(source['_path']) as archive:
        with archive.open(source['member']) as stream:
            for frame in pd.read_csv(stream,header=None,names=['stamp','bid','ask','volume'],dtype={'stamp':str,'bid':float,'ask':float,'volume':float},chunksize=500000):
                stamps=pd.to_datetime(frame.stamp,format='%Y%m%d %H%M%S%f',errors='raise').to_numpy(dtype='datetime64[ms]').astype(np.int64)+5*3600000
                b=frame.bid.to_numpy();a=frame.ask.to_numpy();keys=stamps//86400000
                for day in np.unique(keys):
                    mask=keys==day;date=str(np.datetime64(int(day),'D'));part=folder/(date+'.bin')
                    records=np.empty(int(mask.sum()),dtype=[('time','<i8'),('bid','<f8'),('ask','<f8')]);records['time']=stamps[mask];records['bid']=b[mask];records['ask']=a[mask]
                    with part.open('ab') as out:records.tofile(out)
                    counts[date]=counts.get(date,0)+len(records)
                raw_count+=len(frame)
    for binary in folder.glob('*.bin'):
        rec=np.fromfile(binary,dtype=[('time','<i8'),('bid','<f8'),('ask','<f8')]);order=np.argsort(rec['time'],kind='stable');rec=rec[order]
        np.savez_compressed(binary.with_suffix('.npz'),times=rec['time'],bid=rec['bid'],ask=rec['ask']);binary.unlink()
    result={'source_sha256':source['sha256'],'raw_rows':raw_count,'utc_rows_by_date':counts,'source_timezone':'EST fixed UTC-5','size_fields':'unknown; volume column is not book depth'}
    marker.write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=Path('raw/histdata'));p.add_argument('--cache',type=Path,default=Path('data/histdata_months'));p.add_argument('--result',type=Path,default=Path('results/histdata_year'));p.add_argument('--workers',type=int,default=3);p.add_argument('--offline',action='store_true');args=p.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True);args.result.mkdir(parents=True,exist_ok=True);ledger=args.result/'source_ledger.jsonl'
    started=time.monotonic();finished=0
    with ledger.open('a',buffering=1) as stream:
        def job(spec):
            month,pair=spec
            try:
                result=download(args.raw,month,pair,args.offline)
                with LOCK:stream.write(json.dumps(result)+'\n')
                data=extract({**result,'_path':args.raw/result['archive']},args.cache)
                return {'month':month,'pair':pair,'status':'processed','rows':data['raw_rows'],'bytes':result['bytes']}
            except Exception as exc:
                result={'month':month,'pair':pair,'status':'error','error':type(exc).__name__+': '+str(exc)}
                with LOCK:stream.write(json.dumps(result)+'\n')
                return result
        specs=[(m,pair) for m in MONTHS for pair in PAIRS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(job,spec) for spec in specs]):
                finished+=1;result=future.result();print(json.dumps({**result,'finished':finished,'total':len(specs),'seconds':round(time.monotonic()-started,1)}),flush=True)
                if STOP.is_set():
                    print('STOP: HTTP 429; no further provider requests will be made',flush=True)

if __name__=='__main__':main()
