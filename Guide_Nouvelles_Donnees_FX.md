# Préparer les nouvelles cotations FX

Le moteur complémentaire accepte un nombre variable de devises et de paires, des réseaux incomplets, plusieurs lieux de négociation et des intervalles de dates couvrant des journées entières. Les expériences historiques à quatre devises restent une référence séparée.

## Ce qu'il faut récupérer dès maintenant

| Champ | Usage | Si absent |
|---|---|---|
| Devises de base et de cotation | Sens exact du taux | L'import est refusé |
| Bid et ask | Conversions réellement proposées dans chaque sens | L'import est refusé |
| Horodatage de la cotation, précision et fuseau | Reconstruction du marché | Le fuseau n'est jamais deviné |
| Horodatage de réception locale | Information réellement disponible à la décision | Il est marqué inconnu ; le temps source sert de convention de reconstruction |
| Plateforme / fournisseur / compte accessible | Séparer les marchés et les liquidités | Utiliser un identifiant explicite, même pour un fournisseur d'archives |
| Taille disponible au bid et à l'ask | Limiter les montants | Inconnue, jamais infinie ; le rejeu s'abstient par défaut |
| Identifiant et séquence de mise à jour | Invalidation, doublons, consommation de profondeur | Identifiant de provenance généré à partir de la source et de la ligne |
| Cotation ferme ou indicative, délai de validité | Conditions d'exécution | Fermeté inconnue, aucune promesse de remplissage |
| Frais et financement de l'accès envisagé | Gain net | Un scénario déclaré, pas une estimation du tarif réel |
| Ordres, rejets, exécutions partielles, commissions | Audit des gains exécutés | Les résultats restent des simulations de cotations |

Les tailles bid et ask du format canonique sont exprimées en unités de la devise de base de l'instrument. Pour acheter la base au prix ask, la capacité dans la devise vendue vaut `ask_size_base * ask`. Ne pas confondre volume négocié, compteur de ticks et profondeur disponible.

Le champ `firm=true` ne garantit pas à lui seul l'exécution d'une boucle. Il faut aussi connaître le montant, la validité, les modalités de rejet et les contraintes du compte.

## Organisation conseillée des fichiers

Conserver les fichiers bruts intacts, avec le nom du fournisseur, la paire, les dates, le fuseau et la documentation du format. Le moteur calcule une empreinte SHA-256 de chaque fichier, conserve son interprétation et refuse de réinterpréter silencieusement le même contenu avec une autre convention.

Le modèle `examples/quotes_template.csv` donne les noms canoniques. Il contient uniquement des exemples synthétiques. Le fichier `configs/csv_quotes.json` permet de changer les noms de colonnes. Si chaque CSV correspond à une paire fixe, écrire `base`, `quote` et `venue` comme constantes du JSON et supprimer ces entrées de `columns`.

Pour HistData Generic ASCII Tick Data, `configs/histdata_eurusd.json` montre le format sans en-tête et la convention EST fixe, UTC-5. Dupliquer ce petit JSON par paire. La colonne volume n'est pas utilisée comme taille disponible. Les ZIP bruts sont conservés, et leurs CSV sont extraits avant l'import.

Les dates ISO avec décalage explicite sont préférables. Une valeur d'époque entière exige `timestamp_unit` parmi `s`, `ms`, `us`, `ns`. Les dates locales ambiguës au changement d'heure sont refusées ; utiliser des dates UTC ou un décalage explicite. La précision d'un horodatage ne prouve pas la synchronisation des horloges.

## Import et diagnostics sur journées complètes

Depuis la racine du projet, avec les dépendances du fichier `requirements.txt` :

```bash
python -m fxresearch ingest --config configs/csv_quotes.json --database datasets/fx.sqlite raw/cotations.csv
python -m fxresearch inspect --database datasets/fx.sqlite
python -m fxresearch detect --database datasets/fx.sqlite --config configs/full_days.json --output results/nouvelles_journees
```

Modifier **les dates d'exemple**, la liste des devises et les paramètres dans `configs/full_days.json`. La borne finale est exclusive. Par exemple, du 1er au 31 mars inclus : début `2024-03-01T00:00:00Z`, fin `2024-04-01T00:00:00Z`.

La grille d'exemple est d'une seconde ; elle peut être réglée à 100 ms. Une grille plus espacée peut manquer des opportunités brèves. Le manifeste conserve ce paramètre. Le temps de calcul sur de grands réseaux doit être profilé avant de traiter toute la période.

Le stockage SQLite est indexé, lu dans l'ordre chronologique et n'exige pas une matrice `temps × devises × devises` en mémoire. Les sorties sont écrites au fil de l'eau. Le moteur recalcule les diagnostics quand le graphe change et réutilise le résultat sur les graphes identiques.

Par défaut, seule une paire périmée est retirée du réseau (`freshness_policy: edge`). Une boucle n'utilise donc que les arêtes fraîches encore accessibles. `freshness_policy: all` retrouve le filtre de cohorte commun de l'étude historique. Ces deux populations doivent être comparées séparément.

Les sommets sont `(plateforme, devise)`. Des prix sur deux plateformes ne sont pas reliés par un transfert instantané gratuit. Les transferts doivent être modélisés et financés explicitement ; le rejeu fourni traite les cycles d'une seule plateforme.

## Rejeu hypothétique et audit d'exécutions

```bash
python -m fxresearch replay --database datasets/fx.sqlite --alerts results/nouvelles_journees/alerts.csv --config configs/replay.json --output results/rejeu_nouvelles_journees
python -m fxresearch audit-fills --input mes_ordres_remplis.csv --output results/audit_reel.json --currency USD
```

Le rejeu suit les positions monétaires, les quantités restantes par identifiant de cotation, les frais, le capital, les délais et les échecs. Il interdit les tentatives simultanées sur le même compte. Si une boucle échoue, il tente une liquidation ; si des positions restent ouvertes, le compte simulé est gelé et aucun gain réalisé complet n'est attribué à cette boucle.

Il produit des **ordres hypothétiques**, même si les tailles et le champ de fermeté sont présents. Un modèle de carnet de tête ne reconstruit pas la concurrence, la file d'attente, toutes les annulations ni le last look d'un fournisseur précis. Un ordre FOK sur chaque jambe n'est pas une réservation atomique de toute la boucle.

Par défaut `assumed_size_base` est `null` : les tailles inconnues empêchent l'évaluation de l'exécution. Toute valeur imposée doit être traitée comme un scénario synthétique et apparaît dans le manifeste.

L'audit accepte les flux de cash effectivement exécutés (`examples/fills_template.csv`). Un épisode non fermé conserve ses positions résiduelles et son P&L réalisé est inconnu. Les exécutions dépassant le cash déclaré exigent une réconciliation séparée des crédits/emprunts.

## Découpage statistique à figer avant le test

1. Entraînement : régler les modèles, normalisations et familles de politiques.
2. Calibration : calculer les enveloppes et les bornes de risque sur des blocs complets.
3. Test final : périodes futures jamais utilisées pour choisir un paramètre.
4. Répétitions chronologiques : fenêtres glissantes préannoncées ; aucun ajustement sur le test final.

Les horizons futurs doivent être entièrement observables et purgés aux frontières des périodes. Les séances voisines ne deviennent pas indépendantes par simple changement de date. Les garanties conformes supposent l'échangeabilité des scores de blocs ; les bornes binomiales conditionnelles supposent des blocs indépendants de même loi. Sur le marché, ces hypothèses restent à étudier et ne sont pas vérifiées automatiquement.

Pour une enveloppe conforme à 99 %, il faut au moins **99 blocs de calibration** pour obtenir une borne finie avec la correction de rang exacte. Cela n'inclut ni l'entraînement ni le test. Ce minimum assure seulement la possibilité d'une borne finie, pas sa précision ou son utilité.

Pour une règle de trading sélectionnée parmi plusieurs règles, distinguer : couverture des prix, probabilité de perte par occasion, probabilité de perte parmi les occasions acceptées, et risque d'au moins une perte dans une journée traitée. Le validateur utilise un résultat de Bernoulli par bloc et politique, sans traiter des milliers de ticks corrélés comme des essais indépendants.

```bash
python -m fxresearch calibrate --scores mes_scores_par_jour.json --alpha 0.01 --output results/enveloppe.json
python -m fxresearch validate-policies --input mes_politiques_par_jour.csv --registry configs/frozen_policy_registry.json --alpha 0.05 --delta 0.05 --output results/validation_politiques.json
```

Le registre de politiques est figé avant la calibration et son empreinte est conservée. Les colonnes `accepted` et `loss` du fichier par journée valent 0 ou 1. `loss=1` signifie ici au moins une tentative perdante pendant une journée où la politique agit. Si une politique n'a pas assez de journées acceptées, elle n'est pas validée. On ne remplit pas les résultats de politiques non exécutées par des zéros : les contrefactuels manquants nécessitent une simulation explicitement assumée ou une collecte expérimentale appropriée.

## Vérifier l'installation avant les vraies données

```bash
python -m unittest discover -s tests -v
python demo_full_day.py
```

La démonstration couvre 24 heures, 12 devises, 2 plateformes et une cotation par minute par paire. C'est un essai de traitement sur un flux synthétique clairsemé, pas une validation d'un débit HFT ou de profits de marché.

## Règlement et conventions de la politique

Conserver également le type d’instrument, les dates de valeur, les modalités de règlement, les crédits, marges et règles de compensation. Le modèle suppose des dates compatibles et des soldes réutilisables après chaque conversion ; cela doit être vérifié pour le compte réel.

Le registre exprime ses seuils en points de base de rendement simple. Les alertes contiennent un gain logarithmique : le filtre convertit explicitement `10000 * expm1(gain_bp / 10000)` avant comparaison. Les frais de détection et de rejeu doivent représenter le même scénario.

Pour préparer les phases puis calculer les labels hypothétiques par journée :

```bash
python -m fxresearch split-plan --database datasets/fx.sqlite --output results/plan.json
python -m fxresearch evaluate-policies --database datasets/fx.sqlite --alerts results/nouvelles_journees/alerts.csv --registry configs/frozen_policy_registry.json --config configs/replay.json --output results/politiques_hypothetiques
```

Les labels ainsi produits sont ceux du modèle de rejeu. Ils ne remplacent pas des exécutions réelles contrefactuelles.
