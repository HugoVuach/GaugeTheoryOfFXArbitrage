# Arbitrage FX : huit devises et journées entières

Le rapport conserve ses 50 pages précédentes et ajoute la campagne empirique sur huit devises : USD, EUR, GBP, JPY, CHF, AUD, CAD et NZD, avec les 28 paires directes.

## Données et limites de couverture

- **HistData** : archives Generic ASCII Tick de tous les mois de 2024, avec décembre 2023 comme contexte UTC. Ce sont les prix bid/ask bruts de la campagne annuelle. Les tailles de carnet sont inconnues.
- **Dukascopy** : paquets journaliers bruts effectivement reçus en janvier–mars 2024, avec meilleurs bid/ask et volumes. Le téléchargement annuel prévu a été arrêté sur HTTP 429. Seules les dates dont les 28 paquets sont reçus entrent dans les calculs de quantité ; la liste et les échecs figurent dans le journal de sources.
- La comparaison des sources est effectuée sans les supposer indépendantes. Les boucles ne mélangent pas les fournisseurs et les trous ne sont pas interpolés.
- Tous les remplissages sont des contrefactuels annoncés. Aucun journal d’exécutions réelles n’est créé à partir de prix seuls.

Les archives brutes sont fournies séparément du code pour éviter de dupliquer plusieurs gigaoctets dans le ZIP du projet. Leur index contient les empreintes et les liens. Le code conserve les résultats et observables nécessaires à l’inspection du rapport.

L’index de téléchargement `Donnees_Brutes_FX_8_Devises_2024.md` est inclus dans ce ZIP. Les gros caches dérivés de ticks et de grilles sont omis ; les commandes ci-dessous les reconstruisent. Les tableaux et les six nouvelles figures se régénèrent directement depuis les résultats et les petits échantillons de tracé fournis, sans téléchargement.

## Installation et reproduction hors ligne

Python 3.12 ; dépendances figées dans `requirements.txt`. Une distribution TeX complète avec pdfLaTeX, Latin Modern, babel, PGF/TikZ et tcolorbox est nécessaire pour les figures et le PDF.

```bash
python -m pip install -r requirements.txt
python restore_raw.py --archives /chemin/vers/les_archives --output raw
python collect_histdata_year.py --offline --workers 3
python study_dukascopy.py features --source histdata --raw data/histdata_months --result results/histdata_year
python study_dukascopy.py features --source dukascopy --raw raw/dukascopy --result results/dukascopy
python study_dukascopy.py augment
python study_dukascopy.py replay
python study_dukascopy.py calibrate
python empirical_flow_check.py
python compare_sources.py
python analyse_long_period.py
python report_long_period.py
python make_long_period_report.py
pdflatex -interaction=nonstopmode -halt-on-error Note_Jauge_Arbitrage_FX_3_Experiences.tex
pdflatex -interaction=nonstopmode -halt-on-error Note_Jauge_Arbitrage_FX_3_Experiences.tex
python -m unittest discover -s tests -v
```

La restauration vérifie chaque fichier original avec son SHA-256. Les caches mensuels sont reconstruits depuis les ZIP. Les journaux de sources livrés fixent les dates Dukascopy du corpus. Les commandes d’analyse reprennent les jours déjà entièrement calculés ; déplacer `results/histdata_year/days` et `results/dukascopy/days` pour refaire tous les calculs depuis les originaux, en gardant les journaux de sources.

## Protocole et cibles

`configs/histdata_2024_frozen.json` fixe janvier–mars pour l’apprentissage, avril–septembre pour la calibration et octobre–décembre pour le test annuel. `configs/dukascopy_aux_frozen.json` décrit le sous-protocole de quantité. L’ancien projet annuel Dukascopy est conservé pour documenter le changement imposé par l’accès, sans prétendre avoir reçu les mois manquants.

La grille est de 100 ms, avec le dernier prix source connu à chaque date. La fraîcheur principale est d’une seconde par paire, et les dix dernières secondes de chaque bloc sont purgées pour les labels futurs. Les duplications de temps sont conservées dans les originaux ; la reconstruction utilise le dernier enregistrement à l’heure considérée. Les temps de réception locaux, la fermeté et la file d’attente ne sont pas disponibles.

Les enveloppes sur le **dernier prix archivé** et celles exigeant un **prix futur frais** sont séparées. Un label inconnu devient un score majorant infini ; il n’est pas supprimé après la sélection. Les échelles normalisées sont apprises sur l’entraînement seulement. Les comptes statistiques sont des journées du lundi au vendredi, pas des millions de ticks indépendants.

Le rejeu suit les soldes, volumes consommés, frais et routes. Les mêmes prix/tailles répétés ne reconstituent pas la profondeur. Un inventaire non fermé gèle le compte et garde un P&L complet inconnu. La validation parmi huit seuils vise l’existence d’une perte ou d’un inventaire non fermé pendant une journée traitée : ce critère diffère du signe du P&L quotidien et du taux de pertes par trade.

Les traces enregistrent aussi la devise et le montant résiduels ainsi que les USD déjà récupérés. Le plancher comptable décote à zéro les positions longues restantes et conserve ces recettes partielles ; il n’impute pas un P&L nul aux positions ouvertes. Les variantes supplémentaires du filtre de 100 ms sont un suivi exploratoire documenté dans `configs/robustness_followup.json`, après le premier résumé de test. Les sommes du test portent sur onze comptes journaliers initialisés séparément.

## Fichiers de référence

- `source/previous_50_pages.tex` : rapport précédent autonome conservé.
- `source/long_period.tex` : texte de la campagne empirique.
- `collect_histdata_year.py`, `collect_dukascopy.py` : acquisition traçable ; l’option réseau Dukascopy s’arrête en cas de limitation.
- `study_dukascopy.py`, `fxresearch/empirical.py`, `fxresearch/historical_replay.py` : reconstruction, diagnostics, labels et cash.
- `analyse_long_period.py`, `report_long_period.py` : résumés, comparaisons, tables et figures.
- `results/histdata_year/`, `results/dukascopy/`, `results/source_comparison/` : résultats, sources, audits et traces hypothétiques.
- `README_Legacy.md` : expériences initiales ; les expériences synthétiques restent explicitement séparées des archives réelles.

Le LaTeX final intègre les tableaux et les figures vectorielles et peut se compiler seul. Les propriétés de géométrie, de dualité et de calibration n’impliquent pas une rentabilité ni une priorité scientifique générale.

Le PDF livré compte 63 pages : les 50 pages antérieures sont conservées, puis la campagne commence page 51. Les expériences réelles I, II et III commencent respectivement pages 54, 56 et 59. `python audit_empirical_results.py` recoupe les totaux et les journaux de cash publiés ; les 24 tests unitaires portent notamment sur les unités, les capacités, les labels manquants et la séparation entraînement/test.

L’amendement `configs/histdata_2024_calibration_amendment.json` retire de la calibration les dix dates de juin et juillet déjà explorées dans le rapport précédent. Elles restent dans les originaux et les statistiques descriptives. Le test octobre–décembre est conservé.
