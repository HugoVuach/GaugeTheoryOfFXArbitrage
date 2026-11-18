# Données brutes FX — 8 devises et historique 2024

**Devises : USD, EUR, GBP, JPY, CHF, AUD, CAD et NZD. Les 28 paires directes sont conservées.**

Le corpus annuel HistData contient **663,387,055 ticks** sur les journées de 2024, avec horodatages natifs à la milliseconde, bid et ask. Les treize lots ci-dessous conservent les 364 ZIP originaux : douze mois de 2024 et décembre 2023 pour le contexte UTC. Les tailles de carnet ne sont pas fournies par cette source.

Le sous-corpus Dukascopy d'exécution couvre **65 dates communes aux 28 paires**, dont 54 dates du lundi au vendredi, du 2024-01-02 au 2024-03-20. Les archives fournies conservent aussi les paquets reçus de dates incomplètes. Elles comprennent les volumes associés aux meilleurs bid/ask. La collecte annuelle Dukascopy a été arrêtée sur limitation HTTP 429 ; les mois manquants ne sont pas inventés.

Les sources sont analysées séparément. Leur chevauchement est largement identique ; il ne constitue pas une deuxième répétition indépendante. Les données brutes sont réelles, les remplissages d’ordres du rapport restent hypothétiques.

**Taille totale des lots : 5.12 Go.** Les lots peuvent être téléchargés séparément.

## Archives mensuelles HistData

| Lot | Taille (Mo) | Téléchargement |
|---|---:|---|
| HistData_2023_12_contexte.zip | 362.4 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_334c64b89354819188ad5c5eed6b506d&account_id=personal) |
| HistData_2024_01.zip | 382.8 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_d6f8486da2cc81918b5442d6074b1f45&account_id=personal) |
| HistData_2024_02.zip | 298.5 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_39a781ad259081918ea5657713ad066a&account_id=personal) |
| HistData_2024_03.zip | 275.2 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_3bbdaf72782c8191b577e25542067eb6&account_id=personal) |
| HistData_2024_04.zip | 306.9 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_433c30abc8288191a1a48c06c6cdf1cc&account_id=personal) |
| HistData_2024_05.zip | 271.6 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_c33e10e28b1881918c82397778ebd603&account_id=personal) |
| HistData_2024_06.zip | 264.5 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_ab6462cd78c08191a82c19526914533b&account_id=personal) |
| HistData_2024_07.zip | 323.5 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_2322e184e4648191b0c3389ed6fd4134&account_id=personal) |
| HistData_2024_08.zip | 450.4 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_40c897fd2a7881919fe3eae229f95c5f&account_id=personal) |
| HistData_2024_09.zip | 412.1 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_e10695e729d081918d7e82f375d399f0&account_id=personal) |
| HistData_2024_10.zip | 388.8 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_6a05c7b840148191b812dc8eac663c76&account_id=personal) |
| HistData_2024_11.zip | 463.6 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_bf2aa2d74e4881918631d974ac5d82dc&account_id=personal) |
| HistData_2024_12.zip | 383.3 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_4c1bea4b424081918e6c91880f0484bf&account_id=personal) |

## Paquets Dukascopy effectivement reçus

| Lot | Taille (Mo) | Téléchargement |
|---|---:|---|
| Dukascopy_2024_01_fichiers_recus.zip | 230.0 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_09a8b7c12d088191871569f410b1a186&account_id=personal) |
| Dukascopy_2024_02_fichiers_recus.zip | 186.6 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_75a4e03c80588191b463e24035bae47c&account_id=personal) |
| Dukascopy_2024_03_fichiers_recus.zip | 123.4 | [Ouvrir le fichier](https://chatgpt.com/library?file_id=libfile_67dfbe932fb081919e654a8985fa9148&account_id=personal) |

## Vérification et reproduction

Chaque lot contient un `manifest.csv` avec les URL et SHA-256 de ses membres originaux. Les empreintes des lots sont dans `data/raw_archive_index.csv` du ZIP de code. Les archives sont conservées sans réécriture des données ; leurs noms locaux sont normalisés pour identifier le mois et la paire.

Après téléchargement, la commande `python restore_raw.py --archives DOSSIER_DES_LOTS --output raw` restaure les originaux et vérifie leurs empreintes. Le README du code décrit ensuite la reconstruction hors ligne des caches, les expériences, les figures et le PDF.

HistData est décodé en EST fixe (UTC−5), puis converti en UTC. Dukascopy est décodé au temps source UTC. Aucun temps de réception local, champ de fermeté ou ordre réellement exécuté n’est inféré à partir de ces seules archives.
