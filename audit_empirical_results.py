"""Check released summaries against the hypothetical order cash ledgers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def audit(root=Path('.')):
    reports={}
    for name,expected in [('histdata_year',314),('dukascopy',65)]:
        folder=root/'results'/name
        summary=json.loads((folder/'dataset_summary.json').read_text())
        hourly=pd.read_csv(folder/'hourly.csv')
        quality=pd.read_csv(folder/'quality.csv')
        assert hourly.date.nunique()==summary['dates']==expected
        assert int(hourly.grid.sum())==summary['grid_points']
        assert int(quality.raw_ticks.sum())==summary['raw_ticks']
        assert int(quality.invalid.sum())==summary['invalid_ticks']
        assert int(quality.valid_ticks.sum())==summary['valid_ticks']
        assert quality.groupby('date').pair.nunique().eq(28).all()
        reports[name]={'dates':expected,'raw_ticks':summary['raw_ticks'],
                       'phase_weekdays':summary['phase_weekdays']}
    result=root/'results/dukascopy'
    daily=pd.read_csv(result/'daily_policies.csv')
    assert daily.date.nunique()==65 and daily.method.nunique()==26
    assert len(daily)==65*26 and daily.replay_schema_version.eq(2).all()
    count=0
    for row in daily.itertuples():
        path=result/'replay'/f'{row.date}_{row.method}_attempts.csv.gz'
        if not row.accepted:
            assert not path.exists(),str(path)
            continue
        f=pd.read_csv(path)
        assert len(f)==row.accepted,str(path)
        closed=f.closed.astype(bool)
        assert int((~closed).sum())==row.unclosed
        assert f.loc[~closed,'pnl_usd'].isna().all()
        assert f.loc[~closed,'residual_amount'].gt(0).all()
        assert int((f.loc[closed,'pnl_usd'] < -1e-8).sum())==row.losses
        assert f.episode.nunique()==len(f)
        if len(f)>1:
            assert np.all(f.timestamp_ms.to_numpy()[1:]>=f.end_ms.to_numpy()[:-1])
        if (~closed).any():
            assert (~closed).sum()==1 and not closed.iloc[-1]
        known=float(f.pnl_usd.sum())
        floor=known+float((f.loc[~closed,'terminal_usd_received']-row.notional_usd).sum())
        assert abs(known-row.known_pnl_usd)<1e-5
        assert abs(floor-row.conservative_pnl_usd)<1e-5
        assert abs(f.cash_usd.iloc[-1]-10*row.notional_usd-floor)<1e-5
        assert f.hypothetical.all()
        count+=len(f)
    cash=pd.read_csv(result/'cash_test.csv').set_index('method')
    test=daily[(daily.phase=='test')&(daily.weekday<5)]
    for method,g in test.groupby('method'):
        assert len(g)==11
        for field in ['accepted','unclosed','losses','complete','known_pnl_usd','conservative_pnl_usd']:
            assert abs(float(g[field].sum())-float(cash.loc[method,field]))<1e-5
    annual=pd.read_csv(root/'results/histdata_year/calibration.csv')
    assert annual.calibration_days.eq(121).all()
    test_prices=pd.read_csv(root/'results/histdata_year/price_test.csv')
    assert test_prices.test_days.eq(66).all()
    reports['cash_ledgers']={'daily_policy_blocks':len(daily),'attempts_checked':count,
                            'methods':26,'test_days':11,'status':'all checks passed'}
    (root/'results/release_data_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()
