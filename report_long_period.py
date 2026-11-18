"""Vector figures and generated tables for the full-day empirical supplement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_report import save,TEAL,RUST,BLUE,SLATE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/long_period'
LABELS={'pooled_archived':'Quantile des points','edge_archived':'Bloc / arêtes','joint_archived':'Bloc / trajet','scaled_joint_archived':'Bloc normalisé'}

def integer(value):return f'{int(value):,}'.replace(',',r'\,')
def number(value,digits=2):
    if pd.isna(value):return '--'
    if np.isinf(value):return r'\(+\infty\)'
    return f'{float(value):,.{digits}f}'.replace(',',r'\,').replace('.',',')
def percent(value,digits=1):
    return '--' if pd.isna(value) else number(100*value,digits)+r'\,\%'
def table(path,head,rows,columns):
    text='\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+' & '.join(head)+r'\\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    path.write_text(text)

def figures():
    main=ROOT/'results/histdata_year';aux=ROOT/'results/dukascopy';dest=ROOT/'figures'
    hs=pd.read_csv(main/'hourly.csv');ds=pd.read_csv(aux/'hourly.csv')
    q=pd.read_csv(main/'quality.csv');q['month']=q.date.str[:7]
    fig,ax=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    monthly=q.groupby('month').raw_ticks.sum()/1e6
    ax[0].bar(np.arange(len(monthly)),monthly,color=TEAL);ax[0].set_xticks(np.arange(len(monthly)),[x[5:] for x in monthly.index]);ax[0].set(xlabel='Mois de 2024',ylabel='Ticks bruts (millions)');ax[0].set_title('A Archives annuelles',loc='left')
    for frame,color,label in [(hs,TEAL,'HistData / année'),(ds,RUST,'Dukascopy / dates reçues')]:
        hourly=frame.groupby('hour')[['full_28_fresh','any_pair_points','grid']].sum()
        ax[1].plot(hourly.index,100*hourly.full_28_fresh/hourly.grid,color=color,label=label)
    ax[1].set(xlabel='Heure UTC',ylabel='Grille avec 28 paires fraîches (%)',xlim=(0,23));ax[1].set_title('B Cohorte commune à 1 seconde',loc='left');ax[1].legend(fontsize=7)
    for a in ax:a.grid(axis='y');a.set_axisbelow(True)
    save(fig,dest,'emp_collecte')

    fig,ax=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    fees=pd.read_csv(main/'fees.csv');x=np.arange(len(fees));den=fees.active_grid_points.iloc[0]
    for field,color,offset,label in [('global_positive_frames',TEAL,-.25,'Tous cycles'),('triangle_positive_frames',BLUE,0,'Triangles'),('global_positive_age100_frames',RUST,.25,'Tous cycles / âge 100 ms')]:
        ax[0].bar(x+offset,100*fees[field]/den,width=.24,color=color,label=label)
    ax[0].set_xticks(x,[number(f,1) for f in fees.fee_bp]);ax[0].set(xlabel='Frais ajoutés par conversion (pb)',ylabel='Grilles positives / grilles actives (%)');ax[0].legend(fontsize=7);ax[0].set_title('A Coûts et fraîcheur',loc='left')
    grouped=hs.groupby('date')[['positive_0','any_pair_points']].sum();dates=pd.to_datetime(grouped.index)
    ax[1].plot(dates,100*grouped.positive_0/grouped.any_pair_points,color=TEAL,lw=.9)
    ax[1].axvspan(pd.Timestamp('2024-10-01'),pd.Timestamp('2024-12-31'),color=BLUE,alpha=.09,label='Test final')
    ax[1].set(ylabel='Grilles positives sans frais (%)',xlabel='Journée UTC');ax[1].set_title('B Variations entre journées',loc='left');ax[1].legend(fontsize=7)
    for a in ax:a.grid(axis='y');a.set_axisbelow(True)
    fig.autofmt_xdate();save(fig,dest,'emp_detection')

    sample_files=sorted((main/'days').glob('*_sample10s.npz'))
    if len(sample_files)==json.loads((main/'dataset_summary.json').read_text())['dates']:
        samples=[]
        for path in sample_files:
            with np.load(path) as z:s=z['sample'];samples.append(s[np.isfinite(s[:,4])][::5])
        sample=np.concatenate(samples);np.savez_compressed(main/'plot_geometry_sample.npz',sample=sample)
    else:
        with np.load(main/'plot_geometry_sample.npz') as z:sample=z['sample']
    fig,ax=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    density=ax[0].hexbin(sample[:,4],sample[:,1],gridsize=45,bins='log',mincnt=1,cmap='viridis')
    bar=fig.colorbar(density,ax=ax[0],label='Nombre de points',shrink=.8)
    if bar.solids is not None:bar.solids.set_rasterized(False)
    ax[0].set(xlabel='Résidu de Hodge RMS (pb)',ylabel='Moyenne maximale du graphe (pb)');ax[0].set_title('A Cohorte complète / échantillon',loc='left')
    event_files=sorted((main/'days').glob('*_events.npz'))
    if len(event_files)==json.loads((main/'dataset_summary.json').read_text())['dates']:
        lengths=np.zeros(7,np.int64)
        for path in event_files:
            with np.load(path) as z:
                for L in range(2,9):lengths[L-2]+=int((z['lengths']==L).sum())
        pd.DataFrame({'length':np.arange(2,9),'frames':lengths}).to_csv(main/'route_lengths.csv',index=False)
    else:lengths=pd.read_csv(main/'route_lengths.csv').frames.to_numpy()
    ax[1].bar(np.arange(2,9),100*lengths/max(1,lengths.sum()),color=TEAL);ax[1].set(xlabel='Conversions de la route USD',ylabel='Candidats positifs (%)',xticks=np.arange(2,9));ax[1].set_title('B Longueur choisie par la règle',loc='left')
    save(fig,dest,'emp_geometrie')

    price=pd.read_csv(main/'price_test.csv');sub=price[(price.alpha==.05)&(price.delay_ms==100)].set_index('method').loc[list(LABELS)]
    fig,ax=plt.subplots(1,2,figsize=(8,3.3),layout='constrained');x=np.arange(4)
    ax[0].bar(x-.17,100*sub.point_coverage,.32,color=BLUE,label='Couverture des points');ax[0].bar(x+.17,100*sub.block_coverage,.32,color=TEAL,label='Couverture des journées');ax[0].axhline(95,color=SLATE,ls='--',lw=1)
    ax[0].set(ylabel='Couverture sur le test (%)',ylim=(0,103));ax[0].set_title('A Niveau 95 %, délai 100 ms',loc='left');ax[0].legend(fontsize=7,frameon=True,facecolor='white',edgecolor='none',framealpha=.95,loc='lower left')
    ax[1].bar(x,sub.mean_radius_bp,color=[SLATE,BLUE,TEAL,RUST]);ax[1].set(ylabel='Rayon moyen (pb par conversion)');ax[1].set_title('B Largeur des bornes',loc='left')
    for a in ax:a.set_xticks(x,['Points','Bloc\narêtes','Bloc\ntrajet','Bloc\nnormalisé']);a.grid(axis='y');a.set_axisbelow(True)
    save(fig,dest,'emp_calibration')

    cash=pd.read_csv(aux/'cash_test.csv').set_index('method');methods=['fee_0','threshold_0','latency_500','age_100ms','joint_archived','notional_1m']
    sub=cash.loc[methods];fig,ax=plt.subplots(1,2,figsize=(8,3.3),layout='constrained');x=np.arange(len(methods));den=sub.accepted.replace(0,np.nan)
    ax[0].bar(x-.17,100*sub.complete/den,.32,color=TEAL,label='Route complète');ax[0].bar(x+.17,100*sub.unclosed/den,.32,color=RUST,label='Inventaire non fermé')
    ax[0].set(ylabel='Part des tentatives (%)',ylim=(0,105));ax[0].set_title('A Rejeu avec tailles publiées',loc='left');ax[0].legend(fontsize=7)
    values=sub.known_pnl_usd
    ax[1].bar(x,values,color=np.where(values>=0,TEAL,RUST));ax[1].axhline(0,color=SLATE,lw=.7)
    ax[1].set(ylabel='Flux nets des épisodes fermés (USD)');ax[1].set_title('B Résultats hypothétiques / test',loc='left')
    for a in ax:
        a.set_xticks(x,['Sans\nfrais','Base','Délai\n500 ms','Âge\n100 ms','Borne\nprix','Montant\n1 M']);a.grid(axis='y');a.set_axisbelow(True)
        for i,n in enumerate(sub.accepted):
            if not n:a.text(i,.02,'0 ordre',rotation=90,ha='center',va='bottom',transform=a.get_xaxis_transform(),fontsize=7)
    save(fig,dest,'emp_execution')

    validation=json.loads((aux/'policy_validation.json').read_text());policies=pd.DataFrame(validation['policies'])
    fig,ax=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    thresholds=[float(m.split('_')[1]) for m in policies.method]
    ax[0].plot(thresholds,100*policies.risk_upper,'o-',color=RUST,ms=4);ax[0].axhline(5,color=SLATE,ls='--');ax[0].set(xlabel='Seuil de rendement simple (pb)',ylabel='Borne de risque conditionnel (%)',ylim=(0,104));ax[0].set_title('A Huit politiques figées',loc='left')
    n=np.arange(1,151);u=1-(.05/8)**(1/n);ax[1].plot(n,100*u,color=TEAL,label='Borne si zéro perte');ax[1].axhline(5,color=SLATE,ls='--');ax[1].axvline(policies.traded_calibration_days.max(),color=RUST,ls=':',label='Effectif maximal disponible')
    ax[1].set(xlabel='Journées acceptées indépendantes',ylabel='Meilleure borne à zéro perte (%)',ylim=(0,50));ax[1].set_title('B Résolution statistique',loc='left');ax[1].legend(fontsize=7)
    for a in ax:a.grid(axis='y')
    save(fig,dest,'emp_risque')

def tables():
    OUT.mkdir(parents=True,exist_ok=True)
    h=json.loads((ROOT/'results/histdata_year/dataset_summary.json').read_text());d=json.loads((ROOT/'results/dukascopy/dataset_summary.json').read_text());overlap=json.loads((ROOT/'results/source_comparison/summary.json').read_text())
    table(OUT/'table_sources.tex',['Corpus','Dates','Ticks bruts','Tailles'],[
       ['HistData / 2024',integer(h['dates']),integer(h['raw_ticks']),'Inconnues'],['Dukascopy / reçus',integer(d['dates']),integer(d['raw_ticks']),'Meilleur bid/ask']], 'lrrl')
    fees=pd.read_csv(ROOT/'results/histdata_year/fees.csv');table(OUT/'table_fees.tex',['Frais / jambe','Tous cycles','Triangles','Sans triangle'],[
      [number(r.fee_bp,1),integer(r.global_positive_frames),integer(r.triangle_positive_frames),integer(r.global_positive_without_triangle_frames)] for r in fees.itertuples()], 'lrrr')
    price=pd.read_csv(ROOT/'results/histdata_year/price_test.csv');rows=[]
    for alpha in [.1,.05,.01]:
        for method in LABELS:
            r=price[(price.alpha==alpha)&(price.delay_ms==100)&(price.method==method)].iloc[0]
            rows.append([str(round(100*(1-alpha)))+r'\,\%',LABELS[method],number(r.mean_radius_bp,3),number(100*r.block_coverage,1)+r'\,\%',integer(r.accepted_frames)])
    table(OUT/'table_price.tex',['Niveau','Méthode','Rayon moyen','Couv. blocs','Acceptées'],rows,'llrrr')
    rows=[]
    for delay in [50,100,500]:
        for method in ['pooled_archived','joint_archived','scaled_joint_archived']:
            r=price[(price.alpha==.05)&(price.delay_ms==delay)&(price.method==method)].iloc[0]
            rows.append([str(delay),LABELS[method],integer(r.accepted_frames),percent(r.loss_fraction_given_accepted),integer(r.strict_future_unavailable)])
    table(OUT/'table_selection.tex',['Délai (ms)','Méthode','Acceptées','Perte de prix','Futur non frais'],rows,'llrrr')
    cash=pd.read_csv(ROOT/'results/dukascopy/cash_test.csv').set_index('method');rows=[]
    names={'fee_0':'Frais nuls','threshold_0':'Base : 100 ms, 0,1 pb','latency_50':'Délai 50 ms','latency_500':'Délai 500 ms','age_100ms':'Filtre âge 100 ms','joint_archived':'Borne de prix / bloc','notional_10k':'Montant 10 000 USD','notional_1m':'Montant 1 M USD','validated_policy':'Politique validée'}
    for method,label in names.items():
        r=cash.loc[method];rows.append([label,integer(r.accepted),integer(r.losses),integer(r.unclosed),number(r.known_pnl_usd,0)])
    table(OUT/'table_cash.tex',['Règle','Tentatives','Pertes','Non fermées','Flux fermés USD'],rows,'lrrrr')
    robust=[('age_100ms',.1,100),('age100_fee_0.2',.2,100),('age100_fee_0.5',.5,100),('age100_latency_50',.1,50),('age100_latency_500',.1,500)]
    table(OUT/'table_robustness.tex',['Frais (pb)','Délai (ms)','Tentatives','Non fermées','Plancher cash USD'],[
      [number(fee,1),str(delay),integer(cash.loc[name].accepted),integer(cash.loc[name].unclosed),number(cash.loc[name].conservative_pnl_usd,0)] for name,fee,delay in robust],'lrrrr')
    validation=json.loads((ROOT/'results/dukascopy/policy_validation.json').read_text());table(OUT/'table_risk.tex',['Seuil (pb)','Jours traités','Perte ou inconnu','Borne du risque'],[
       [p['method'].split('_')[1],integer(p['traded_calibration_days']),integer(p['losing_or_unknown_days']),number(100*p['risk_upper'],1)+r'\,\%'] for p in validation['policies']], 'lrrr')
    lp=pd.read_csv(ROOT/'results/dukascopy/actual_coefficients_lp.csv')
    macros={'HistTicks':integer(h['raw_ticks']),'DukaTicks':integer(d['raw_ticks']),'HistDates':integer(h['dates']),'DukaDates':integer(d['dates']),'HistWeekdays':integer(h['weekday_dates']),'DukaWeekdays':integer(d['weekday_dates']),
      'GridPoints':integer(h['grid_points']),'ActivePoints':integer(h['active_grid_points']),'FreshAllPercent':number(100*h['full_28_fresh_points']/h['grid_points'],2),'HistInvalid':integer(h['invalid_ticks']),'DukaInvalid':integer(d['invalid_ticks']),
      'HistWholeSeconds':number(100*h['whole_second_fraction'],2),'DukaWholeSeconds':number(100*d['whole_second_fraction'],2),'HistDuplicates':number(100*h['duplicate_timestamp_fraction'],2),'DukaDuplicates':number(100*d['duplicate_timestamp_fraction'],2),
      'MatchedPairDays':integer(overlap['identical_price_pair_days']),'ComparedPairDays':integer(overlap['pair_days_compared']),'MatchedDates':integer(overlap['dates']),
      'HistTrain':integer(h['phase_weekdays']['training']),'HistCal':integer(h['phase_weekdays']['calibration']),'HistTest':integer(h['phase_weekdays']['test']),
      'DukaTrain':integer(d['phase_weekdays']['training']),'DukaCal':integer(d['phase_weekdays']['calibration']),'DukaTest':integer(d['phase_weekdays']['test']),
      'HistCandidateFrames':integer(h['candidate_frames']),'HistEpisodes':integer(h['candidate_episodes']),'GlobalPeak':number(h['global_max_mean_bp'],3),
      'LPCount':integer(len(lp)),'LPGap':f'{lp.duality_gap_usd.abs().max():.2e}','LPUnitError':f'{lp.gauge_error_usd.abs().max():.2e}'}
    for alpha,tag in [(.05,'NinetyFive'),(.01,'NinetyNine')]:
        r=price[(price.alpha==alpha)&(price.delay_ms==100)&(price.method=='joint_archived')].iloc[0]
        macros[tag+'Radius']=number(r.mean_radius_bp,3);macros[tag+'Coverage']=number(100*r.block_coverage,1);macros[tag+'Accepted']=integer(r.accepted_frames)
    pooled=price[(price.alpha==.05)&(price.delay_ms==100)&(price.method=='pooled_archived')].iloc[0]
    scaled=price[(price.alpha==.05)&(price.delay_ms==100)&(price.method=='scaled_joint_archived')].iloc[0]
    macros.update(PooledPointCoverage=number(100*pooled.point_coverage,2),PooledBlockCoverage=number(100*pooled.block_coverage,1),PooledAccept=integer(pooled.accepted_frames),PooledLosses=integer(pooled.losing_quote_frames),PooledUnfresh=integer(pooled.strict_future_unavailable),ScaledRadius=number(scaled.mean_radius_bp,3),ScaledCoverage=number(100*scaled.block_coverage,2),ScaledAccept=integer(scaled.accepted_frames))
    for name,prefix in [('threshold_0','Base'),('age_100ms','Age')]:
        r=cash.loc[name]
        macros.update({prefix+'Attempts':integer(r.accepted),prefix+'Losses':integer(r.losses),prefix+'Open':integer(r.unclosed),prefix+'ClosedCash':number(r.known_pnl_usd,2),prefix+'Floor':number(r.conservative_pnl_usd,2)})
    (OUT/'values.tex').write_text('\n'.join('\\newcommand{\\Emp'+key+'}{'+value+'}' for key,value in macros.items())+'\n')

if __name__=='__main__':
    figures();tables()
