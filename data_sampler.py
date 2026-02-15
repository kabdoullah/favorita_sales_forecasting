import pandas as pd
import numpy as np
from typing import Literal, Optional


class SalesDataSampler:
    """
    Classe pour échantillonner intelligemment des données de ventes.

    Stratégies disponibles:
    1. Random sampling: Échantillonnage aléatoire simple
    2. Stratified sampling: Échantillonnage stratifié par caractéristiques
    3. Time-based sampling: Échantillonnage temporel (périodes récentes ou fenêtre glissante)
    4. Store-based sampling: Échantillonnage par magasins entiers
    5. Item-based sampling: Échantillonnage par familles/catégories de produits
    6. Hybrid sampling: Combinaison de plusieurs méthodes
    """

    def __init__(self, train_df: pd.DataFrame, items_df: Optional[pd.DataFrame] = None):
        """
        Initialise le sampler.

        Args:
            train_df: DataFrame principal avec colonnes [date, store_nbr, item_nbr, unit_sales, ...]
            items_df: DataFrame optionnel avec informations sur les items (family, class, perishable)
        """
        self.train = train_df.copy()
        self.items = items_df.copy() if items_df is not None else None

        if not pd.api.types.is_datetime64_any_dtype(self.train['date']):
            self.train['date'] = pd.to_datetime(self.train['date'])

        self.n_rows = len(self.train)
        self.n_stores = self.train['store_nbr'].nunique()
        self.n_items = self.train['item_nbr'].nunique()
        self.date_min = self.train['date'].min()
        self.date_max = self.train['date'].max()
        self.n_days = (self.date_max - self.date_min).days + 1

    def random_sample(self, frac: float = 0.1, random_state: int = 42) -> pd.DataFrame:
        """
        Échantillonnage aléatoire simple.

        AVANTAGES:
        - Rapide et simple
        - Préserve les proportions globales

        INCONVÉNIENTS:
        - Brise la continuité temporelle (crucial pour forecast!)
        - Perd la structure des séries temporelles
        - Peu adapté pour la prévision

        QUAND UTILISER:
        - Exploration rapide des données
        - Tests de code
        - PAS pour l'entraînement de modèles de prévision!

        Args:
            frac: Fraction à échantillonner (0.0-1.0)
            random_state: Graine aléatoire pour reproductibilité

        Returns:
            DataFrame échantillonné
        """
        sample = self.train.sample(frac=frac, random_state=random_state)
        return sample.sort_values(['date', 'store_nbr', 'item_nbr']).reset_index(drop=True)

    def time_based_sample(
        self,
        method: Literal['recent', 'window', 'periodic'] = 'recent',
        n_days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Échantillonnage temporel - RECOMMANDÉ pour time series!

        AVANTAGES:
        - Préserve la continuité temporelle (CRUCIAL pour forecast)
        - Cohérent avec la logique métier (prévision sur données récentes)
        - Permet de simuler la production (train sur passé, test sur futur)

        INCONVÉNIENTS:
        - Peut perdre des patterns saisonniers si période trop courte
        - Sensible aux événements exceptionnels dans la période choisie

        QUAND UTILISER:
        - Modélisation (FORTEMENT RECOMMANDÉ)
        - Feature engineering
        - Validation temporelle (time series split)

        Méthodes:
        - 'recent': N derniers jours
        - 'window': Fenêtre de dates spécifique [start_date, end_date]
        - 'periodic': Échantillonnage périodique (ex: 1 semaine sur 4)

        Args:
            method: Méthode d'échantillonnage temporel
            n_days: Nombre de jours (pour 'recent')
            start_date: Date de début (pour 'window'), format 'YYYY-MM-DD'
            end_date: Date de fin (pour 'window'), format 'YYYY-MM-DD'
            period_freq: Fréquence périodique (pour 'periodic'), ex: '7D', '1M'

        Returns:
            DataFrame échantillonné
        """
        if method == 'recent':
            if n_days is None:
                raise ValueError("Spécifiez n_days pour la méthode 'recent'")
            cutoff_date = self.date_max - pd.Timedelta(days=n_days)
            sample = self.train[self.train['date'] >= cutoff_date].copy()

        elif method == 'window':
            if start_date is None or end_date is None:
                raise ValueError("Spécifiez start_date et end_date pour la méthode 'window'")
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            sample = self.train[(self.train['date'] >= start) & (self.train['date'] <= end)].copy()

        elif method == 'periodic':
            if period_freq is None:
                raise ValueError("Spécifiez period_freq pour la méthode 'periodic'")
            dates = pd.date_range(self.date_min, self.date_max, freq=period_freq)
            sample = self.train[self.train['date'].isin(dates)].copy()

        else:
            raise ValueError(f"Méthode inconnue: {method}")

        return sample.reset_index(drop=True)

    def store_based_sample(self, frac: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """
        Échantillonnage par magasins entiers - EXCELLENT pour développement!

        AVANTAGES:
        - Préserve TOUTES les séries temporelles des magasins sélectionnés
        - Permet d'analyser la dynamique complète magasin par magasin
        - Idéal pour feature engineering (lags, rolling stats, etc.)
        - Résultats généralisables aux autres magasins

        INCONVÉNIENTS:
        - Ne capture pas la diversité de tous les magasins
        - Taille fixe par magasin (certains plus gros que d'autres)

        QUAND UTILISER:
        - Développement et tests (TRÈS RECOMMANDÉ)
        - Feature engineering
        - Modèles par magasin (store-level models)

        Args:
            frac: Fraction de magasins à échantillonner (0.0-1.0)
            random_state: Graine aléatoire

        Returns:
            DataFrame échantillonné
        """
        all_stores = self.train['store_nbr'].unique()
        n_stores_sample = max(1, int(len(all_stores) * frac))

        np.random.seed(random_state)
        selected_stores = np.random.choice(all_stores, size=n_stores_sample, replace=False)

        sample = self.train[self.train['store_nbr'].isin(selected_stores)].copy()
        return sample.sort_values(['date', 'store_nbr', 'item_nbr']).reset_index(drop=True)

    def item_based_sample(
        self,
        method: Literal['random', 'by_family'] = 'by_family',
        frac: float = 0.2,
        families: Optional[list] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Échantillonnage par items/familles de produits.

        AVANTAGES:
        - Permet de se concentrer sur certaines catégories
        - Réduit la dimensionnalité (moins d'items à prévoir)
        - Préserve la continuité temporelle par item

        INCONVÉNIENTS:
        - Perd des informations cross-items
        - Certaines familles peuvent être plus difficiles à prévoir

        QUAND UTILISER:
        - Analyse par catégorie de produits
        - Modèles spécialisés par famille
        - Tests de modèles sur sous-ensembles homogènes

        Args:
            method: 'random' (items aléatoires) ou 'by_family' (familles spécifiques)
            frac: Fraction d'items à échantillonner (pour 'random')
            families: Liste de familles à inclure (pour 'by_family')
            random_state: Graine aléatoire

        Returns:
            DataFrame échantillonné
        """
        if method == 'random':
            all_items = self.train['item_nbr'].unique()
            n_items_sample = max(1, int(len(all_items) * frac))

            np.random.seed(random_state)
            selected_items = np.random.choice(all_items, size=n_items_sample, replace=False)
            sample = self.train[self.train['item_nbr'].isin(selected_items)].copy()

        elif method == 'by_family':
            if self.items is None:
                raise ValueError("Fournissez items_df pour utiliser 'by_family'")
            if families is None:
                raise ValueError("Spécifiez une liste de familles pour 'by_family'")

            selected_items = self.items[self.items['family'].isin(families)]['item_nbr'].unique()
            sample = self.train[self.train['item_nbr'].isin(selected_items)].copy()

        else:
            raise ValueError(f"Méthode inconnue: {method}")

        return sample.sort_values(['date', 'store_nbr', 'item_nbr']).reset_index(drop=True)

    def stratified_sample(
        self,
        frac: float = 0.1,
        stratify_by: Literal['store', 'date', 'store_date'] = 'store_date',
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Échantillonnage stratifié.

        AVANTAGES:
        - Préserve les proportions par strate
        - Garantit la représentativité

        INCONVÉNIENTS:
        - Brise la continuité temporelle (sauf si stratifié par date)
        - Plus lent que random sampling

        QUAND UTILISER:
        - Exploration avec garantie de représentativité
        - Tests statistiques nécessitant des proportions préservées

        Args:
            frac: Fraction à échantillonner
            stratify_by: Dimension de stratification
            random_state: Graine aléatoire

        Returns:
            DataFrame échantillonné
        """
        if stratify_by == 'store':
            sample = self.train.groupby('store_nbr', group_keys=False).apply(
                lambda x: x.sample(frac=frac, random_state=random_state)
            )
        elif stratify_by == 'date':
            sample = self.train.groupby('date', group_keys=False).apply(
                lambda x: x.sample(frac=frac, random_state=random_state)
            )
        elif stratify_by == 'store_date':
            sample = self.train.groupby(['store_nbr', 'date'], group_keys=False).apply(
                lambda x: x.sample(frac=min(frac, 1.0), random_state=random_state)
            )
        else:
            raise ValueError(f"stratify_by inconnu: {stratify_by}")

        return sample.sort_values(['date', 'store_nbr', 'item_nbr']).reset_index(drop=True)

    def hybrid_sample(
        self,
        store_frac: float = 0.3,
        recent_days: int = 365,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Échantillonnage hybride: Combine stores + période récente.

        AVANTAGES:
        - Meilleur compromis entre réduction de taille et préservation de structure
        - Préserve continuité temporelle + cohérence par magasin
        - Focalise sur données récentes (plus pertinentes pour prévision)

        RECOMMANDATION PRINCIPALE POUR CE PROJET!

        Args:
            store_frac: Fraction de magasins (ex: 0.3 = 30% des magasins)
            recent_days: Nombre de jours récents à conserver
            random_state: Graine aléatoire

        Returns:
            DataFrame échantillonné
        """
        all_stores = self.train['store_nbr'].unique()
        n_stores_sample = max(1, int(len(all_stores) * store_frac))

        np.random.seed(random_state)
        selected_stores = np.random.choice(all_stores, size=n_stores_sample, replace=False)

        cutoff_date = self.date_max - pd.Timedelta(days=recent_days)

        sample = self.train[
            (self.train['store_nbr'].isin(selected_stores)) &
            (self.train['date'] >= cutoff_date)
        ].copy()

        return sample.sort_values(['date', 'store_nbr', 'item_nbr']).reset_index(drop=True)

    def get_sample_stats(self, sample_df: pd.DataFrame) -> dict:
        """
        Calcule des statistiques sur un échantillon.

        Args:
            sample_df: DataFrame échantillonné

        Returns:
            Dictionnaire de statistiques
        """
        return {
            'n_rows': len(sample_df),
            'n_stores': sample_df['store_nbr'].nunique(),
            'n_items': sample_df['item_nbr'].nunique(),
            'date_min': sample_df['date'].min(),
            'date_max': sample_df['date'].max(),
            'n_days': (sample_df['date'].max() - sample_df['date'].min()).days + 1,
            'pct_of_original': len(sample_df) / self.n_rows * 100,
            'mean_sales': sample_df['unit_sales'].mean(),
            'median_sales': sample_df['unit_sales'].median(),
            'total_sales': sample_df['unit_sales'].sum()
        }

    def compare_samples(self, sample_df: pd.DataFrame) -> None:
        """
        Compare un échantillon avec les données originales.

        Args:
            sample_df: DataFrame échantillonné
        """
        print("\n" + "="*70)
        print("COMPARAISON ÉCHANTILLON vs DONNÉES ORIGINALES")
        print("="*70)

        stats = self.get_sample_stats(sample_df)

        print(f"\nTaille:")
        print(f"  Original: {self.n_rows:,} lignes")
        print(f"  Échantillon: {stats['n_rows']:,} lignes ({stats['pct_of_original']:.2f}%)")

        print(f"\nMagasins:")
        print(f"  Original: {self.n_stores}")
        print(f"  Échantillon: {stats['n_stores']} ({stats['n_stores']/self.n_stores*100:.1f}%)")

        print(f"\nItems:")
        print(f"  Original: {self.n_items}")
        print(f"  Échantillon: {stats['n_items']} ({stats['n_items']/self.n_items*100:.1f}%)")

        print(f"\nPériode:")
        print(f"  Original: {self.date_min.date()} → {self.date_max.date()} ({self.n_days} jours)")
        print(f"  Échantillon: {stats['date_min'].date()} → {stats['date_max'].date()} ({stats['n_days']} jours)")

        print(f"\nVentes (unit_sales):")
        print(f"  Original - Moyenne: {self.train['unit_sales'].mean():.2f}, Médiane: {self.train['unit_sales'].median():.2f}")
        print(f"  Échantillon - Moyenne: {stats['mean_sales']:.2f}, Médiane: {stats['median_sales']:.2f}")

        print("="*70)

    def save_sample(
        self,
        sample_df: pd.DataFrame,
        method_name: str,
        output_dir: str = 'data/samples',
        optimize: bool = True
    ) -> str:
        """
        Sauvegarde un échantillon avec un nom descriptif.

        Args:
            sample_df: DataFrame à sauvegarder
            method_name: Nom de la méthode d'échantillonnage
            output_dir: Répertoire de sortie
            optimize: Optimiser les types avant sauvegarde

        Returns:
            Chemin du fichier sauvegardé
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        if optimize:
            sample_df = optimize_dtypes(sample_df)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"train_sample_{method_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        sample_df.to_csv(filepath, index=False)
        print(f"✓ Échantillon sauvegardé: {filepath}")

        return filepath


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise les types de données pour réduire l'empreinte mémoire.

    Args:
        df: DataFrame à optimiser

    Returns:
        DataFrame avec types optimisés
    """
    df_optimized = df.copy()

    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

    float_cols = df_optimized.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

    for col in df_optimized.select_dtypes(include=['object']).columns:
        if col != 'date':
            n_unique = df_optimized[col].nunique()
            n_total = len(df_optimized)
            if n_unique / n_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')

    return df_optimized
