"""Generate and process 24h of SPARSE SYNTHETIC quotes, 12 currencies, 2 venues.

One quote per minute per pair: an ingestion/causality smoke test, not a dense
high-frequency market simulation or an empirical financial result.
"""
from pathlib import Path
import csv
import json
import argparse
import time
import numpy as np
from fxresearch.data import ingest_csv, iso_utc, timestamp_ns
from fxresearch.pipeline import run_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('demo_generated'))
    args=parser.parse_args();root=args.directory;root.mkdir(parents=True,exist_ok=True)
    symbols=['USD','EUR','GBP','JPY','CHF','AUD','NZD','CAD','SEK','NOK','SGD','HKD']
    rng=np.random.default_rng(131801)
    values=np.r_[0.,np.log([1.09,1.27,1/150,1.12,.66,.61,.73,.095,.094,.75,.128])*10000]
    start=timestamp_ns('2024-08-01T00:00:00Z');began=time.perf_counter()
    path=root/'synthetic_quotes.csv'
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['timestamp_market_utc','timestamp_receive_utc','venue','base','quote','bid','ask','bid_size_base','ask_size_base','quote_id','firm'])
        for minute in range(1440):
            values[1:]+=rng.normal(0,.15,len(symbols)-1)
            stamp=start+minute*60*1000000000
            for venue in ['SYNTH_A','SYNTH_B']:
                for i in range(len(symbols)):
                    for j in range(i+1,len(symbols)):
                        mid=values[i]-values[j]
                        writer.writerow([iso_utc(stamp),iso_utc(stamp+10000000),venue,symbols[i],symbols[j],
                                         np.exp((mid-.1)/10000),np.exp((mid+.1)/10000),'','',f'{venue}_{minute}_{i}_{j}',''])
    cfg=json.loads(Path('configs/csv_quotes.json').read_text())
    ingest=ingest_csv(path,root/'quotes.sqlite',cfg)
    config={'currencies':symbols,'start_utc':iso_utc(start+20000000),
            'end_utc':iso_utc(start+86400*1000000000+20000000),
            'step_ms':60000,'max_age_ms':1000,'freshness_policy':'edge','compute_triangles':True}
    result=run_diagnostics(root/'quotes.sqlite',config,root/'diagnostics')
    summary={'source':'sparse synthetic plumbing test','currencies':len(symbols),'venues':2,'rows':ingest['rows'],
             'full_session_hours':24,'quote_spacing_seconds':60,'decisions':result['grid_points'],
             'episodes':result['episodes'],'end_to_end_seconds':time.perf_counter()-began,
             'sizes':'missing by design; execution must be unassessable','manifest':result}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
