# Dossier des Données

Ce dossier doit contenir les fichiers de données du projet Favorita Sales Forecasting.

## 📥 Téléchargement des Données

Les fichiers de données ne sont **pas inclus dans le dépôt Git** en raison de leur taille (4,7 GB au total).

### Méthode 1 : Kaggle (Recommandée)

1. Créer un compte sur [Kaggle](https://www.kaggle.com/)
2. Aller sur : https://www.kaggle.com/c/favorita-grocery-sales-forecasting
3. Accepter les règles de la compétition
4. Télécharger les fichiers depuis l'onglet "Data"
5. Extraire tous les fichiers CSV dans ce dossier

### Méthode 2 : Kaggle CLI

```bash
pip install kaggle
kaggle competitions download -c favorita-grocery-sales-forecasting
unzip favorita-grocery-sales-forecasting.zip -d .
```

## 📁 Fichiers Requis

Après téléchargement, ce dossier doit contenir :

```
data/
├── train.csv              ✅ 4,7 GB - Données d'entraînement (2013-2017)
├── test.csv               ✅ 120 MB - Données de test
├── items.csv              ✅ 102 KB - Métadonnées des articles
├── stores.csv             ✅ 1,4 KB - Informations sur les magasins
├── oil.csv                ✅ 20 KB - Prix quotidiens du pétrole
├── holidays_events.csv    ✅ 22 KB - Calendrier des jours fériés
└── transactions.csv       ✅ 1,5 MB - Nombre de transactions quotidiennes
```

Les petits fichiers (items.csv, stores.csv, oil.csv, holidays_events.csv, transactions.csv) sont déjà inclus dans le dépôt Git.

## ⚠️ Note Importante

Seuls les gros fichiers **train.csv** et **test.csv** doivent être téléchargés depuis Kaggle. Les autres fichiers sont déjà présents dans ce dossier.
