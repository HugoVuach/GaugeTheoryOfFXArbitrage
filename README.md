# Jauge, holonomie et arbitrage — trois expériences

Le rapport développe les connexions sur les fibrés en droites, les périodes de de Rham et la décomposition de Hodge sur le réseau des devises. Il comprend une partie complète pour chacune des trois expériences supplémentaires.

## Expériences et données

1. experiences.py construit un cylindre à huit sommets, douze arêtes et quatre faces. Il teste 19 481 scénarios de période et de spread et vérifie le seuil analytique.
2. Le même script compare trois enveloppes temporelles sur six paires HistData entre EUR, GBP, USD et JPY. Les cinq fenêtres du 10 au 14 juin 2024 calibrent les rayons. Les cinq fenêtres du 8 au 12 juillet évaluent les règles. La grille va de 12 h 20 à 13 h 00 UTC à pas de 100 ms. Les trois quantiles testés sont 90 %, 95 % et 99 %.
3. Il simule quarante trajectoires indépendantes d'un marché martingale, quinze scénarios d'observation et quatre délais d'exécution. Les paramètres d'actualisation sont des intensités nominales. Sur la grille, l'attente moyenne effective est indiquée dans le rapport.

Le dossier results contient les résultats numériques des expériences, les validations et les résultats historiques de la version précédente. Le fichier data/exp2_cycle_panels.npz conserve les gains moyens nominaux et rejoués des vingt cycles, les masques de fraîcheur et les erreurs maximales des douze directions. Il permet de recalculer l'expérience 2 sans télécharger les cotations. Ce fichier est un cache d'observables dérivées.

Les fichiers source/note.tex et source/experiences.tex contiennent le texte éditable. Le script experience_report.py produit les figures, tableaux et valeurs des trois expériences. Le script make_report.py assemble un LaTeX autonome avec les figures vectorielles et les tableaux intégrés.

## Reproduire les trois expériences

Environnement testé avec Python 3.12 et les versions de requirements.txt. Une distribution TeX complète avec pdfLaTeX, Latin Modern, babel, TikZ, tcolorbox et latexmk sert aux figures PGF et à la compilation.

~~~bash
python -m pip install -r requirements.txt
python experiences.py
python make_report.py
latexmk -pdf -interaction=nonstopmode -halt-on-error Note_Jauge_Arbitrage_FX_3_Experiences.tex
~~~

Les graines, paramètres et quantiles du protocole sont fixés dans le code. Les textes chiffrés du rapport décrivent ce protocole par défaut. Les tableaux et figures sont reconstruits depuis les résultats produits.

Le fichier Note_Jauge_Arbitrage_FX_3_Experiences.tex se compile également seul dans un dossier vide. Avec pdfLaTeX, lancer deux compilations successives pour résoudre les renvois.

## Reconstruire le cache depuis les données de marché

Les douze archives Generic ASCII Tick Data de juin et juillet 2024 représentent environ 168 Mo. Les pages du fournisseur et les empreintes exactes sont conservées dans results/sources_sha256.csv. Le fournisseur utilise l'EST fixe, UTC−5, que le code convertit en UTC.

~~~bash
python experiences.py --download --refresh-data
~~~

Les fichiers sont téléchargés dans raw/. Si le formulaire du fournisseur change, ouvrir les URL du manifeste et y placer les ZIP sous les noms de la colonne archive. Le script vérifie chaque empreinte avant de reconstruire le cache. L'option --raw CHEMIN permet de choisir un autre dossier.

Pour recalculer aussi les résultats historiques antérieurs aux trois expériences, exécuter python analyse.py, puis python make_report.py. La commande python gauge.py lance les vérifications mathématiques de base. Les expériences ajoutent les contrôles sur le cylindre, les bornes de cycle et les changements de jauge dépendant du temps.

## Lecture des unités et des résultats

Les petits gains sont en points de base logarithmiques. La référence temporelle est un capital détenu dans la devise de départ avec portage nul dans les unités de cotation. L'expérience 2 mesure des gains aux cotations archivées, avec un filtre de fraîcheur pour le rejeu. L'expérience 3 exécute aux prix synchrones du modèle aux instants de conversion. Ses fréquences dépendent des paramètres synthétiques déclarés.

Les couvertures de l'expérience 2 sont descriptives, avec une calibration sur juin et un test sur juillet. Les intervalles de l'expérience 3 utilisent les quarante trajectoires indépendantes comme unités de répétition.
