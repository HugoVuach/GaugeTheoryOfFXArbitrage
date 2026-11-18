"""Vector figures and generated numbers for the appended research report."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_report import (INK,TEAL,RUST,SLATE,GRID,PALE,BLUE,save,decimal,integer)

LABELS={'marginal_observation':'Marginal / observation','fixed_action_block':'Action fixée / bloc',
        'joint_block':'Conjoint / bloc','scaled_joint_block':'Conjoint normalisé'}
METHODS=['nominal','age_100ms','price_certificate','execution_certificate','conditional_risk_policy']
SHORT=['Nominal','Âge 100 ms','Borne prix','Borne cash','Risque conditionnel']


def build_figures(result,figures):
    figures.mkdir(exist_ok=True,parents=True)
    data=pd.read_csv(result/'exp4_bounds.csv')
    fig,axs=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for column,color,label in [('box_radius_bp',SLATE,'Boîte brute'),('projected_box_radius_bp',BLUE,'Boîte après projection'),('factor_radius_bp',TEAL,'Support conjoint exact')]:
        sorted_values=np.sort(data[column])
        axs[0].plot(np.linspace(0,1,len(data)),sorted_values,color=color,label=label,lw=1.7)
    axs[0].set(xlabel='Quantile des 2 000 cas',ylabel='Rayon total (pb log)')
    axs[0].set_title('A Incertitude sur le trajet',loc='left');axs[0].legend(loc='upper left',fontsize=7)
    counts=[data.box_accept.sum(),data.projected_box_accept.sum(),data.factor_accept.sum()]
    axs[1].bar([0,1,2],counts,color=[SLATE,BLUE,TEAL],width=.65)
    axs[1].set_xticks([0,1,2],['Boîte\nbrute','Boîte\nprojetée','Support\nconjoint'])
    axs[1].set(ylabel='Acceptations avec capacité valide',ylim=(0,1600));axs[1].set_title('B Même ensemble de chocs',loc='left')
    for i,v in enumerate(counts):axs[1].text(i,v+28,str(v),ha='center',fontweight='bold')
    for ax in axs:ax.grid(axis='y');ax.set_axisbelow(True)
    save(fig,figures,'extension_bornes')

    data=pd.read_csv(result/'exp5_methods.csv')
    fig,axs=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for regime,color,offset,label in [('exchangeable',TEAL,-.18,'Même loi des blocs'),('volatility_shift',RUST,.18,'Volatilité multipliée par 2,5')]:
        sub=data.query('regime==@regime').set_index('method').loc[list(LABELS)]
        x=np.arange(4)+offset
        axs[0].bar(x,100*sub.block_coverage,.34,color=color,label=label)
        axs[1].bar(x,100*sub.bad_block_rate,.34,color=color,label=label)
    for ax in axs:
        ax.set_xticks(range(4),['Marginal','Action\nfixée','Conjoint','Conjoint\nnormalisé']);ax.grid(axis='y');ax.set_axisbelow(True)
    axs[0].set(ylabel='Couverture du bloc entier (%)',ylim=(0,105));axs[0].axhline(95,color=SLATE,ls='--',lw=1)
    axs[1].set(ylabel='Blocs avec acceptation erronée (%)',ylim=(0,105));axs[1].axhline(5,color=SLATE,ls='--',lw=1)
    axs[0].set_title('A Couverture jointe',loc='left');axs[1].set_title('B Risque après sélection',loc='left')
    axs[0].legend(loc='lower right',fontsize=7)
    save(fig,figures,'extension_couverture')

    online=pd.read_csv(result/'exp5_online.csv')
    fig,axs=plt.subplots(2,1,figsize=(8,3.8),layout='constrained',sharex=True)
    axs[0].plot(online.block,online.score,color=PALE,lw=.45,label='Score du bloc')
    axs[0].plot(online.block,online.adaptive_radius,color=TEAL,lw=1.4,label='ACI, rayon en ligne')
    axs[0].plot(online.block,online.fixed_radius,color=RUST,ls='--',lw=1.2,label='Rayon figé')
    axs[0].set(ylabel='Rayon (pb log)');axs[0].legend(ncol=3,fontsize=7,loc='upper left')
    for column,color,label in [('fixed_error',RUST,'Figé'),('adaptive_error',TEAL,'ACI')]:
        axs[1].plot(online.block,100*online[column].rolling(200,min_periods=200).mean(),color=color,label=label)
    axs[1].axhline(5,color=SLATE,ls='--',lw=1)
    axs[1].set(xlabel='Bloc indépendant (rupture au bloc 1 500)',ylabel='Erreurs sur 200 blocs (%)',ylim=(0,105))
    for ax in axs:ax.axvline(1500,color=SLATE,ls=':');ax.grid(axis='y');ax.set_axisbelow(True)
    save(fig,figures,'extension_adaptation')

    data=pd.read_csv(result/'exp6_methods.csv')
    fig,axs=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
    for mode,color,offset,label in [('reserved',TEAL,-.18,'Réservation complète'),('unreserved',RUST,.18,'Sans réservation')]:
        sub=data.query('mode==@mode').set_index('method').loc[METHODS]
        y=sub.mean_return_per_opportunity_bp.to_numpy()
        err=np.vstack([y-sub.mean_ci_low_bp,sub.mean_ci_high_bp-y])
        axs[0].bar(np.arange(5)+offset,y,.34,yerr=err,color=color,capsize=2,label=label)
        vals=100*sub.loss_fraction_if_accepted.to_numpy()
        axs[1].bar(np.arange(5)+offset,np.nan_to_num(vals),.34,color=color,label=label)
        for i,v in enumerate(vals):
            if np.isnan(v):axs[1].text(i+offset,2,'0 ordre',ha='center',rotation=90,fontsize=6.5)
    for ax in axs:
        ax.set_xticks(range(5),['Nominal','Âge\n100 ms','Borne\nprix','Borne\ncash','Risque\nconditionnel'])
        ax.grid(axis='y');ax.set_axisbelow(True)
    axs[0].set_title('A Résultat net du simulateur',loc='left');axs[0].set(ylabel='Gain par occasion (pb simples)')
    axs[0].legend(fontsize=7,loc='upper right')
    axs[1].set_title('B Pertes parmi les acceptées',loc='left');axs[1].set(ylabel='Part des tentatives acceptées (%)',ylim=(0,105))
    axs[1].axhline(5,color=SLATE,ls='--',lw=1)
    save(fig,figures,'extension_execution')

    policies=pd.read_csv(result/'exp6_conditional_policies.csv')
    fig,axs=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for mode,color,label in [('reserved',TEAL,'Réservation complète'),('unreserved',RUST,'Sans réservation')]:
        sub=policies.query('mode==@mode')
        axs[0].plot(sub.threshold_bp,100*sub.conditional_risk_upper,'o-',color=color,label=label)
        axs[1].plot(sub.threshold_bp,sub.calibration_accepted,'o-',color=color,label=label)
    axs[0].axhline(5,color=SLATE,ls='--',lw=1)
    axs[0].set(xlabel='Seuil nominal figé (pb)',ylabel='Borne de risque conditionnel (%)',ylim=(0,105))
    axs[1].set(xlabel='Seuil nominal figé (pb)',ylabel='Acceptations en calibration')
    axs[0].set_title('A Correction sur huit règles',loc='left');axs[1].set_title('B Taille réellement informative',loc='left')
    axs[0].legend(loc='upper left',fontsize=7)
    for ax in axs:ax.grid(axis='y');ax.set_axisbelow(True)
    save(fig,figures,'extension_selection')

    data=pd.read_csv(result/'scalability.csv')
    fig,axs=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for density,style in [(1.,'-'),(.2,'--')]:
        sub=data.query('density_target==@density')
        for column,color,label in [('bf_median_ms',TEAL,'Bellman--Ford'),('karp_median_ms',BLUE,'Karp')]:
            axs[0].plot(sub.currencies,sub[column],marker='o',ls=style,color=color,label=f'{label}, densité {density:g}')
    axs[0].set(xscale='log',yscale='log',xlabel='Nombre de devises',ylabel='Temps médian (ms)')
    axs[0].set_xticks([4,8,16,32,64],['4','8','16','32','64']);axs[0].legend(fontsize=6.5)
    n=np.array([4,8,16,32,64])
    axs[1].plot(n,8*(n+1)*n/1024,'o-',color=TEAL,label='Table dynamique de Karp')
    axs[1].set(xlabel='Nombre de devises',ylabel='Stockage théorique de la table (Kio)')
    axs[0].set_title('A Machine de cette expérience',loc='left');axs[1].set_title('B Aucune dimension temps dense',loc='left')
    for ax in axs:ax.grid(axis='y');ax.set_axisbelow(True)
    save(fig,figures,'extension_echelle')


def build_tables(result):
    data=pd.read_csv(result/'exp4_bounds.csv')
    lines=[r'\begin{tabularx}{\linewidth}{@{}Yrr@{}}',r'\toprule',r'Borne & Rayon médian (pb log) & Acceptations\\',r'\midrule']
    for name,radius,accepted in [('Boîte brute','box_radius_bp','box_accept'),('Boîte après projection','projected_box_radius_bp','projected_box_accept'),('Support conjoint exact','factor_radius_bp','factor_accept')]:
        lines.append(f'{name} & {decimal(data[radius].median(),3)} & {integer(data[accepted].sum())}'+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabularx}']);(result/'table_extension_bornes.tex').write_text('\n'.join(lines))
    data=pd.read_csv(result/'exp5_methods.csv')
    for regime,suffix in [('exchangeable','stable'),('volatility_shift','rupture')]:
        lines=[r'\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}',r'\toprule',r'Méthode & Rayon & Couv. bloc & Bloc erroné & Acceptées\\',r'\midrule']
        for _,row in data.query('regime==@regime').iterrows():
            lines.append(f'{LABELS[row["method"]]} & {decimal(row.mean_radius_bp,3)} & {decimal(100*row.block_coverage,2)}'+r'\,\% & '+decimal(100*row.bad_block_rate,2)+r'\,\% & '+integer(row.accepted_attempts)+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabularx}']);(result/f'table_extension_{suffix}.tex').write_text('\n'.join(lines))
    data=pd.read_csv(result/'exp6_methods.csv')
    for mode,suffix in [('reserved','reserve'),('unreserved','ouvert')]:
        lines=[r'\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}',r'\toprule',r'Règle & Acceptées & Pertes & Pertes / acceptées & Gain / occasion\\',r'\midrule']
        sub=data.query('mode==@mode').set_index('method')
        for name,label in zip(METHODS,SHORT):
            row=sub.loc[name]
            fraction='--' if pd.isna(row.loss_fraction_if_accepted) else decimal(100*row.loss_fraction_if_accepted,2)+r'\,\%'
            lines.append(f'{label} & {integer(row.accepted)} & {integer(row.losses)} & {fraction} & {decimal(row.mean_return_per_opportunity_bp,3)}'+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabularx}']);(result/f'table_extension_{suffix}.tex').write_text('\n'.join(lines))
    four=json.loads((result/'exp4_summary.json').read_text());five=json.loads((result/'exp5_summary.json').read_text())
    six=json.loads((result/'exp6_protocol.json').read_text())
    vals={'ExtFactorAccept':integer(four['factor_accepts']),'ExtProjectedAccept':integer(four['projected_box_accepts']),
          'ExtCapacityReject':integer(four['capacity_rejections']),'ExtBoxRatio':decimal(four['median_width_ratio'],2),
          'ExtProjectedRatio':decimal(four['median_projected_ratio'],2),'ExtOnlineFixed':decimal(100*five['online_fixed_error_frequency'],2),
          'ExtOnlineAdaptive':decimal(100*five['online_adaptive_error_frequency'],2),'ExtJointRadius':decimal(five['joint_radius_bp'],3),
          'ExtConditionalThreshold':decimal(six['conditional_policies']['reserved']['threshold_bp'],1),
          'ExtConditionalUpper':decimal(100*six['conditional_policies']['reserved']['risk_upper'],2)}
    for mode,prefix in [('reserved','ExtReserved'),('unreserved','ExtOpen')]:
        row=data.query('mode==@mode and method=="conditional_risk_policy"').iloc[0]
        vals[prefix+'Accept']=integer(row.accepted);vals[prefix+'Loss']=integer(row.losses)
        vals[prefix+'Mean']=decimal(row.mean_return_per_opportunity_bp,3)
    texts=['% Valeurs générées; aucune saisie manuelle des résultats.']
    texts += ['\\newcommand{\\'+name+'}{'+value+'}' for name,value in vals.items()]
    (result/'extension_valeurs.tex').write_text('\n'.join(texts)+'\n')


def run(result=Path('results/extension'),figures=Path('figures')):
    build_figures(result,figures);build_tables(result)


if __name__=='__main__':run()
