"""Audit overlap without assuming that two distributors are independent feeds."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from study_dukascopy import DayTape
from collect_dukascopy import PAIRS

def run():
    raw=Path('raw/dukascopy');cache=Path('data/histdata_months');result=Path('results/source_comparison');result.mkdir(exist_ok=True)
    rows=[]
    for path in sorted(Path('results/dukascopy/days').glob('????-??-??.json')):
        metadata=json.loads(path.read_text());day=metadata['date']
        if metadata.get('status')!='processed':continue
        month=day[:7].replace('-','')
        if not all((cache/month/p/'complete.json').exists() for p in PAIRS):continue
        d=DayTape(raw,day);h=DayTape(cache,day,'histdata')
        for pair,dp,hp in zip(PAIRS,d.parts,h.parts):
            scale=1000 if pair.endswith('JPY') else 100000
            same_times=np.array_equal(dp[0],hp[0])
            residual=max(float(np.abs(hp[1]*scale-np.rint(hp[1]*scale)).max()) if len(hp[1]) else 0.,float(np.abs(hp[2]*scale-np.rint(hp[2]*scale)).max()) if len(hp[2]) else 0.)
            same_prices=same_times and residual<1e-6 and np.array_equal(np.rint(dp[1]*scale),np.rint(hp[1]*scale)) and np.array_equal(np.rint(dp[2]*scale),np.rint(hp[2]*scale))
            def digest(p):
                canonical=np.empty(len(p[0]),dtype=[('t','<i8'),('bid','<i8'),('ask','<i8')]);canonical['t']=p[0];canonical['bid']=np.rint(p[1]*scale).astype(np.int64);canonical['ask']=np.rint(p[2]*scale).astype(np.int64)
                return hashlib.sha256(canonical.tobytes()).hexdigest()
            rows.append({'date':day,'pair':pair,'dukascopy_rows':len(dp[0]),'histdata_rows':len(hp[0]),'identical_timestamp_sequence':same_times,'identical_tick_price_sequence':same_prices,'max_histdata_off_tick_residual':residual,'dukascopy_price_sha256':digest(dp),'histdata_price_sha256':digest(hp)})
        print('source overlap',day,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(result/'overlap.csv',index=False)
    summary={'pair_days_compared':len(frame),'dates':frame.date.nunique(),'identical_timestamp_pair_days':int(frame.identical_timestamp_sequence.sum()),'identical_price_pair_days':int(frame.identical_tick_price_sequence.sum()),'matched_dukascopy_rows':int(frame.loc[frame.identical_tick_price_sequence,'dukascopy_rows'].sum()),'comparison_resolution':'original integer millisecond timestamps, bid/ask rounded only to native instrument tick units','inference_limit':'agreement on this overlap does not establish independent sources or identity outside the observed overlap'}
    (result/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':run()
