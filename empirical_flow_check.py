"""Ex-post finite LP and unit checks using actual published rates and sizes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from study_dukascopy import DayTape
from fxresearch.historical_replay import quote_at
from fxresearch.flow import CashEdge,funded_flow

def run(raw=Path('raw/dukascopy'),result=Path('results/dukascopy')):
    rng=np.random.default_rng(28082024);rows=[];details=[]
    for path in sorted((result/'days').glob('????-??-??_events.npz')):
        day=path.name[:10];e=np.load(path)
        if not len(e['times']):continue
        tape=DayTape(raw,day);accepted=None
        for candidate in range(min(100,len(e['times']))):
            t=int(e['times'][candidate]);L=int(e['lengths'][candidate]);route=e['routes'][candidate,:L];edges=[]
            for k in range(L):
                rate,capacity,_,_,_=quote_at(t+(k+1)*100,int(route[k]),int(route[(k+1)%L]),*tape.flat)
                if rate<=0 or not np.isfinite(capacity):break
                edges.append(CashEdge(k,k+1,rate*(1-.1/10000),capacity))
            if len(edges)==L:accepted=(candidate,t,L,route,edges);break
        if accepted is None:continue
        candidate,t,L,route,edges=accepted;edges.append(CashEdge(0,L,1.,100000.))
        original=funded_flow(L+1,edges,0,L,100000.)
        units=np.exp(rng.normal(0,.7,L+1))
        changed=[CashEdge(edge.source,edge.destination,edge.rate*units[edge.destination]/units[edge.source],edge.capacity*units[edge.source]) for edge in edges]
        shifted=funded_flow(L+1,changed,0,L,100000*units[0],repayment=units[L]/units[0])
        rows.append({'date':day,'timestamp_ms':t,'length':L,'ex_post_best_terminal_usd':original['terminal_cash'],'ex_post_profit_usd':original['net_profit'],'duality_gap_usd':original['duality_gap'],'gauge_error_usd':shifted['terminal_cash']/units[L]-original['terminal_cash']})
        details.append({'date':day,'candidate':candidate,'route':route.tolist(),'edges':[edge.__dict__ for edge in edges],'initial_amount':100000,'interpretation':'future coefficients known ex post; no reservation or execution claim'})
    pd.DataFrame(rows).to_csv(result/'actual_coefficients_lp.csv',index=False)
    (result/'actual_coefficients_lp_inputs.json').write_text(json.dumps(details,indent=2))
    print('Actual-coefficient LP cases:',len(rows),'max absolute gap',max(abs(r['duality_gap_usd']) for r in rows),'max unit error',max(abs(r['gauge_error_usd']) for r in rows))

if __name__=='__main__':run()
