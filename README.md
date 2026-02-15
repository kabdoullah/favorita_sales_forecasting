# Prévision des Ventes Favorita

Un projet complet de machine learning pour la prévision des ventes de produits d'épicerie au niveau article, utilisant les données du distributeur équatorien Corporacion Favorita. Ce projet implémente des stratégies d'échantillonnage intelligentes, une ingénierie de features avancée et plusieurs modèles ML avec suivi des expériences via MLflow.

## Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Description du Dataset](#description-du-dataset)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Téléchargement des Données](#téléchargement-des-données)
- [Guide d'Utilisation](#guide-dutilisation)
- [Stratégies d'Échantillonnage](#stratégies-déchantillonnage)
- [Ingénierie des Features](#ingénierie-des-features)
- [Modèles et Évaluation](#modèles-et-évaluation)
- [Suivi des Expériences avec MLflow](#suivi-des-expériences-avec-mlflow)
- [Résultats](#résultats)
- [Prochaines Étapes](#prochaines-étapes)

## Vue d'Ensemble

### Objectif Métier

Prévoir les ventes quotidiennes unitaires pour des milliers d'articles vendus dans différents magasins Favorita en Équateur. Une prévision précise permet d'optimiser la gestion des stocks, de réduire le gaspillage (particulièrement pour les produits périssables) et d'améliorer la satisfaction client.

### Défis Clés

- **Dataset Volumineux**: 125 millions de lignes (4,7 GB) nécessitant un traitement efficace en mémoire
- **Granularité Fine**: Prévisions au niveau article (pas seulement par magasin ou catégorie)
- **Séries Temporelles**: Dépendances temporelles et motifs de saisonnalité
- **Événements Spéciaux**: Jours fériés, promotions, tremblements de terre affectant les ventes

### Approche

1. Échantillonnage intelligent des données pour gérer les contraintes mémoire
2. Analyse exploratoire complète des données
3. Ingénierie de features avancée (temporelles, lags, statistiques glissantes)
4. Plusieurs modèles ML (LightGBM, XGBoost, Random Forest)
5. Suivi des expériences MLflow pour la reproductibilité

## Description du Dataset

**Source**: Corporacion Favorita (dataset de compétition Kaggle)

### Fichiers

| Fichier | Description | Taille | Enregistrements |
|---------|-------------|--------|-----------------|
| `train.csv` | Données d'entraînement (2013-2017) | 4,7 GB | 125M lignes |
| `test.csv` | Données de test pour les prédictions | 120 MB | 3,3M lignes |
| `items.csv` | Métadonnées des articles (famille, classe, périssable) | 102 KB | 4 100 articles |
| `stores.csv` | Informations magasins (ville, état, type, cluster) | 1,4 KB | 54 magasins |
| `transactions.csv` | Transactions quotidiennes par magasin | 1,5 MB | 83 488 lignes |
| `oil.csv` | Prix quotidiens du pétrole (indicateur économique) | 20 KB | 1 218 jours |
| `holidays_events.csv` | Jours fériés nationaux et régionaux | 22 KB | 350 événements |

### Statistiques Clés

- **Période**: 1er janvier 2013 au 15 août 2017 (1 687 jours)
- **54 magasins** à travers l'Équateur (17 villes, 16 états)
- **4 036 articles uniques** dans 33 familles de produits
- **Ventes négatives**: 7 795 enregistrements (0,01%) représentant des retours produits
- **Promotions**: 17% des enregistrements ont des données de promotion manquantes

## Structure du Projet

```
favorita_sales_forecasting/
│
├── 01_Exploration.ipynb          # Exploration et compréhension des données
├── 02_Analyse.ipynb              # Patterns de ventes, corrélations, insights
├── 03_Feature_Engineering.ipynb  # Création de features et prétraitement
├── 04_Modelisation.ipynb         # Entraînement et évaluation des modèles
│
├── data_sampler.py               # Utilitaires d'échantillonnage intelligent
├── requirements.txt              # Dépendances Python
│
├── data/
│   ├── train.csv                 # Données brutes d'entraînement
│   ├── test.csv                  # Données brutes de test
│   ├── items.csv                 # Métadonnées articles
│   ├── stores.csv                # Informations magasins
│   ├── oil.csv                   # Prix du pétrole
│   ├── holidays_events.csv       # Calendrier des jours fériés
│   ├── transactions.csv          # Nombre de transactions
│   ├── samples/                  # Datasets échantillonnés
│   └── processed/                # Features traitées
│
├── models/                       # Modèles entraînés
│   ├── lgb_model.txt            # Modèle LightGBM
│   └── xgb_model.pkl            # Modèle XGBoost
│
├── outputs/                      # Visualisations et résultats
├── mlruns/                       # Données de tracking MLflow
└── .claude/                      # Mémoire agent (notes projet)
```

## Installation

### Prérequis

- Python 3.8+
- 8+ GB RAM recommandés (pour travailler avec des échantillons)
- 32+ GB RAM pour le traitement du dataset complet

### Configuration

1. Cloner ou télécharger le projet:

```bash
cd favorita_sales_forecasting
```

2. Créer un environnement virtuel:

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

3. Installer les dépendances:

```bash
pip install -r requirements.txt
```

### Dépendances

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
joblib>=1.3.0
lightgbm>=4.6.0
xgboost>=3.2.0
mlflow>=3.9.0
```

## Téléchargement des Données

⚠️ **Important**: Les fichiers de données volumineux (`train.csv` et `test.csv`) ne sont pas inclus dans ce dépôt Git en raison des limitations de taille de GitHub.

### Option 1 : Téléchargement depuis Kaggle (Recommandé)

1. Créer un compte sur [Kaggle](https://www.kaggle.com/) si vous n'en avez pas
2. Aller sur la page de la compétition : [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
3. Accepter les règles de la compétition
4. Télécharger les fichiers dans l'onglet "Data"
5. Extraire tous les fichiers CSV dans le dossier `data/` du projet

**Avec Kaggle CLI** (plus rapide):

```bash
# Installer Kaggle CLI
pip install kaggle

# Configurer votre API token (voir https://www.kaggle.com/docs/api)
# Télécharger les données
kaggle competitions download -c favorita-grocery-sales-forecasting

# Extraire dans le dossier data/
unzip favorita-grocery-sales-forecasting.zip -d data/
```

### Option 2 : Utiliser les échantillons générés

Si vous voulez juste tester le code sans télécharger 4,7 GB de données, vous pouvez créer des échantillons :

```python
from data_sampler import SalesDataSampler

# Créer un petit échantillon pour tests rapides
sampler = SalesDataSampler('data/train.csv')
sample_df = sampler.hybrid_sample(frac=0.01)  # 1% des données
sampler.save_sample(sample_df, 'test_sample')
```

### Fichiers requis

Après téléchargement, votre dossier `data/` doit contenir :

```
data/
├── train.csv              # 4,7 GB - Données d'entraînement
├── test.csv               # 120 MB - Données de test
├── items.csv              # 102 KB - Métadonnées articles
├── stores.csv             # 1,4 KB - Informations magasins
├── oil.csv                # 20 KB - Prix du pétrole
├── holidays_events.csv    # 22 KB - Jours fériés
└── transactions.csv       # 1,5 MB - Transactions quotidiennes
```

## Guide d'Utilisation

### 1. Exploration des Données (01_Exploration.ipynb)

**Objectif**: Comprendre la structure du dataset, les distributions et les problèmes de qualité.

**Contenu**:
- Vue d'ensemble du dataset et analyse de l'utilisation mémoire
- Stratégie d'échantillonnage hybride (20% magasins × 365 jours récents)
- Évaluation de la qualité des données (valeurs manquantes, outliers, doublons)
- Résumés statistiques de base

**Insight clé**: Le dataset est trop volumineux pour être chargé directement. L'échantillonnage hybride réduit 125M lignes à 5M (réduction de 96%) tout en préservant la continuité temporelle.

### 2. Analyse des Données (02_Analyse.ipynb)

**Objectif**: Découvrir les patterns de ventes, corrélations et insights métier.

**Contenu**:
- Tendances des ventes dans le temps (quotidiennes, hebdomadaires, mensuelles, annuelles)
- Analyse de l'impact des promotions
- Performance par magasin et famille de produits
- Analyse de corrélation entre features
- Effets des jours fériés et événements spéciaux
- Corrélation entre prix du pétrole et ventes

**Insights clés**:
- Forte saisonnalité hebdomadaire (pics le week-end)
- Les promotions augmentent significativement les ventes
- Les articles périssables ont des patterns différents
- Les indicateurs économiques (prix du pétrole) sont corrélés aux ventes

### 3. Ingénierie des Features (03_Feature_Engineering.ipynb)

**Objectif**: Créer des features prédictives à partir des données brutes.

**Features créées**:
- **Temporelles**: Année, mois, jour, jour de semaine, trimestre
- **Cycliques**: Encodages sinus/cosinus pour mois et jour de semaine
- **Calendrier**: Week-end, début/fin de mois, indicateurs jour de paie
- **Features de lag**: Ventes d'il y a 7 et 14 jours
- **Statistiques glissantes**: Moyenne mobile sur 7 jours
- **Économiques**: Prix du pétrole (forward-filled)
- **Événements**: Indicateurs jours fériés, transactions
- **Encodages catégoriels**: Magasin, article, famille, cluster
- **Spécifiques au domaine**: Indicateur périssable, indicateur retour

**Sortie**: Datasets prétraités sauvegardés dans `data/processed/`

### 4. Modélisation (04_Modelisation.ipynb)

**Objectif**: Entraîner et évaluer les modèles de machine learning.

**Modèles entraînés**:
- **LightGBM** (modèle principal)
- **XGBoost** (comparaison)
- **Random Forest** (baseline optionnelle)

**Workflow**:
1. Charger les features traitées
2. Split temporel train-validation 80/20
3. Entraîner les modèles avec early stopping
4. Évaluer avec plusieurs métriques
5. Analyse de l'importance des features
6. Diagnostics des résidus
7. Comparaison des modèles

**Toutes les expériences sont trackées avec MLflow**

## Stratégies d'Échantillonnage

Le module `data_sampler.py` fournit la classe `SalesDataSampler` avec six stratégies d'échantillonnage:

### 1. Échantillonnage Aléatoire
- **Cas d'usage**: Tests rapides de code
- **Avantages**: Rapide, simple
- **Inconvénients**: Brise la continuité temporelle (MAUVAIS pour séries temporelles)

### 2. Échantillonnage Temporel (RECOMMANDÉ pour modélisation)
- **Méthodes**:
  - `recent`: N derniers jours
  - `window`: Plage de dates spécifique
  - `periodic`: Chaque Nième période
- **Avantages**: Préserve la structure temporelle
- **Inconvénients**: Peut manquer des patterns saisonniers si période trop courte

### 3. Échantillonnage par Magasin
- **Cas d'usage**: Développement et feature engineering
- **Avantages**: Séries temporelles complètes par magasin
- **Inconvénients**: Peut ne pas représenter tous les types de magasins

### 4. Échantillonnage par Article
- **Cas d'usage**: Analyse par catégorie de produits
- **Avantages**: Historiques complets des articles
- **Inconvénients**: Applicabilité limitée

### 5. Échantillonnage Stratifié
- **Cas d'usage**: Représentation équilibrée
- **Avantages**: Maintient les proportions par catégorie
- **Inconvénients**: Complexe, brise la continuité temporelle

### 6. Échantillonnage Hybride (MEILLEUR pour exploration)
- **Combine**: Sélection de magasins + période temporelle récente
- **Exemple**: 20% des magasins × 365 derniers jours
- **Avantages**: Meilleur équilibre entre taille, structure et pertinence
- **Configurations recommandées**:
  - Exploration rapide: 10% magasins × 180 jours (1-2% des données)
  - Développement: 20% magasins × 365 jours (4-5% des données)
  - Feature engineering: 30% magasins × 365 jours (7-10% des données)

### Exemple d'Utilisation

```python
from data_sampler import SalesDataSampler, optimize_dtypes
import pandas as pd

# Charger le dataset complet
train = pd.read_csv('data/train.csv', parse_dates=['date'])
items = pd.read_csv('data/items.csv')

# Initialiser le sampler
sampler = SalesDataSampler(train, items)

# Créer échantillon hybride (RECOMMANDÉ)
sample = sampler.hybrid_sample(
    store_frac=0.2,    # 20% des magasins
    recent_days=365,   # 365 derniers jours
    random_state=42    # Pour reproductibilité
)

# Optimiser l'utilisation mémoire
sample = optimize_dtypes(sample)

# Comparer échantillon avec original
sampler.compare_samples(sample)

# Sauvegarder pour réutilisation
sampler.save_sample(sample, 'hybrid_20pct_365d')
```

## Ingénierie des Features

### Catégories de Features

1. **Features Temporelles** (12 features)
   - Basiques: année, mois, jour, jour_semaine, jour_année, semaine_année, trimestre
   - Calendrier: est_weekend, est_début_mois, est_fin_mois, est_jour_paie
   - Cycliques: mois_sin, mois_cos, jour_semaine_sin, jour_semaine_cos

2. **Features de Lag** (3 features)
   - sales_lag_7: Ventes d'il y a 7 jours
   - sales_lag_14: Ventes d'il y a 14 jours
   - sales_rolling_mean_7: Moyenne mobile sur 7 jours

3. **Données Externes** (3 features)
   - dcoilwtico: Prix quotidien du pétrole
   - is_holiday: Indicateur jour férié
   - transactions: Nombre de transactions quotidiennes

4. **Encodages Catégoriels** (8 features)
   - Encodage fréquence: store_nbr_freq, item_nbr_freq
   - Encodage label: city, state, store_type, family, class, cluster

5. **Spécifiques au Domaine** (4 features)
   - onpromotion: Indicateur promotion
   - perishable: Indicateur article périssable
   - is_return: Ventes négatives (retours)
   - unit_sales_abs: Valeur absolue des ventes

**Total**: 35 features pour l'entraînement du modèle

### Optimisation Mémoire

La fonction `optimize_dtypes()` réduit l'utilisation mémoire de 40-50%:
- int64 → int8/int16/int32 (basé sur la plage de valeurs)
- float64 → float32 (précision suffisante)
- Réduction typique: 5,6 GB → 2,8 GB

## Modèles et Évaluation

### Modèles

#### 1. LightGBM (Modèle Principal)

**Pourquoi LightGBM**:
- Excellent pour grands datasets (entraînement rapide)
- Gère les valeurs manquantes nativement
- Conçu pour le gradient boosting sur arbres
- Performance supérieure sur données tabulaires

**Configuration**:
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_boost_round': 1000,
    'early_stopping_rounds': 50
}
```

#### 2. XGBoost (Comparaison)

**Pourquoi XGBoost**:
- Standard industriel pour les compétitions
- Régularisation robuste
- Bon pour comparaison avec LightGBM

#### 3. Random Forest (Optionnel)

**Pourquoi Random Forest**:
- Baseline simple
- Moins sujet au surapprentissage
- Plus lent sur grands datasets

### Métriques d'Évaluation

1. **RMSE** (Root Mean Squared Error)
   - Métrique de régression standard
   - Pénalise fortement les grandes erreurs

2. **MAE** (Mean Absolute Error)
   - Erreur de prédiction absolue moyenne
   - Plus interprétable que RMSE

3. **Score R2** (Coefficient de Détermination)
   - Proportion de variance expliquée
   - Plage: -∞ à 1 (1 = parfait)

4. **NWRMSLE** (Normalized Weighted Root Mean Squared Logarithmic Error)
   - Métrique officielle de la compétition Kaggle
   - Poids: Articles périssables × 1,25, autres × 1,0
   - Transformation logarithmique réduit l'impact des grandes valeurs
   - Pénalise plus les sous-estimations que les sur-estimations

**Formule**:
```
NWRMSLE = sqrt(Σ(poids × (log(y_pred + 1) - log(y_true + 1))²) / Σ(poids))
```

### Split Train-Validation

**Stratégie**: Split temporel (80/20)
- Train: 80% premiers de la période
- Validation: 20% derniers de la période
- **Pas de mélange** (préserve l'ordre temporel)

**Pourquoi split temporel**:
- Simule le scénario de production (prévoir le futur à partir du passé)
- Prévient la fuite de données du futur vers le passé
- Teste la capacité du modèle à généraliser sur périodes temporelles inconnues

## Suivi des Expériences avec MLflow

### Configuration

MLflow est configuré pour logger toutes les expériences localement:

```python
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("favorita-sales-forecasting")
```

### Ce qui est Loggé

Pour chaque exécution de modèle, MLflow tracke:

1. **Paramètres**:
   - Hyperparamètres du modèle (learning_rate, num_leaves, etc.)
   - Configuration du dataset (train_size, val_size, n_features)
   - Paramètres d'entraînement (early_stopping_rounds, num_boost_round)

2. **Métriques**:
   - Métriques d'entraînement: train_rmse, train_mae, train_r2, train_nwrmsle
   - Métriques de validation: val_rmse, val_mae, val_r2, val_nwrmsle
   - Meilleure itération (pour early stopping)

3. **Artefacts**:
   - Modèles entraînés (fichiers modèles)
   - Visualisations (importance des features, graphiques de résidus)
   - Tableaux de résultats (CSV de comparaison)

4. **Tags**:
   - model_type: "LightGBM", "XGBoost", "RandomForest"
   - dataset: Stratégie d'échantillonnage utilisée

### Visualisation des Résultats

Lancer l'interface MLflow:

```bash
mlflow ui
```

Puis ouvrir: **http://localhost:5000**

**Fonctionnalités**:
- Comparer plusieurs exécutions côte à côte
- Filtrer les exécutions par métriques (ex: `metrics.val_nwrmsle < 0.5`)
- Télécharger les modèles entraînés
- Voir les visualisations et artefacts
- Suivre l'historique des expériences

### Charger un Modèle Loggé

```python
import mlflow.lightgbm

# Charger modèle depuis une exécution spécifique
logged_model = 'runs:/f59a3f4098094ab7beb85d685a9bd9d6/model'
model = mlflow.lightgbm.load_model(logged_model)

# Faire des prédictions
predictions = model.predict(X_test)
```

## Résultats

### Dataset Utilisé pour l'Entraînement

- **Stratégie d'échantillonnage**: Hybride (20% magasins × 365 jours récents)
- **Total échantillons**: 6 980 883 enregistrements
- **Échantillons entraînement**: 5 584 706 (80%)
- **Échantillons validation**: 1 396 177 (20%)
- **Features**: 35 features ingéniérées

### Performance des Modèles

#### LightGBM (Modèle Principal)

| Métrique | Entraînement | Validation |
|----------|--------------|------------|
| **RMSE** | 4,77 | 14,26 |
| **MAE** | 0,11 | 0,26 |
| **R2** | 0,943 | 0,772 |
| **NWRMSLE** | 0,032 | 0,056 |

- **Meilleure itération**: 180 (sur 1000)
- **Early stopping**: Déclenché à l'itération 230

#### XGBoost (Comparaison)

| Métrique | Validation |
|----------|------------|
| **RMSE** | 22,95 |
| **MAE** | 1,54 |
| **R2** | 0,411 |
| **NWRMSLE** | 0,234 |

### Gagnant: LightGBM

LightGBM surpasse significativement XGBoost sur toutes les métriques:
- **75% meilleur NWRMSLE** (0,056 vs 0,234)
- **37% meilleur RMSE** (14,26 vs 22,95)
- **83% meilleur MAE** (0,26 vs 1,54)

### Top 20 Features les Plus Importantes (LightGBM)

1. `sales_rolling_mean_7` - Moyenne mobile 7 jours
2. `sales_lag_7` - Ventes d'il y a 7 jours
3. `sales_lag_14` - Ventes d'il y a 14 jours
4. `item_nbr_freq` - Encodage fréquence article
5. `dayofweek` - Jour de la semaine (0=Lundi)
6. `transactions` - Nombre de transactions quotidiennes
7. `store_nbr_freq` - Encodage fréquence magasin
8. `month` - Mois de l'année
9. `onpromotion` - Indicateur promotion
10. `dcoilwtico` - Prix du pétrole
11. `family_encoded` - Famille de produits
12. `cluster` - Cluster du magasin
13. `perishable` - Indicateur article périssable
14. `dayofyear` - Jour de l'année
15. `unit_sales_abs` - Ventes absolues
16. `year` - Année
17. `is_weekend` - Indicateur week-end
18. `class_encoded` - Classe de l'article
19. `city_encoded` - Ville du magasin
20. `type_encoded` - Type de magasin

**Insight clé**: Les features de lag et statistiques glissantes sont les prédicteurs les plus puissants, confirmant la forte autocorrélation temporelle dans les données de ventes.

### Diagnostics du Modèle

**Analyse des Résidus**:
- Les résidus sont approximativement distribués normalement
- Légère hétéroscédasticité (variance augmente avec la magnitude de prédiction)
- Quelques outliers subsistent (événements de ventes extrêmes)
- Résidu moyen ≈ 0 (prédictions non biaisées)

**Axes d'Amélioration**:
- Sous-estimation pour les événements de très hautes ventes
- Pourrait bénéficier de plus de features de lag (21, 28 jours)
- Les méthodes d'ensemble pourraient réduire la variance
