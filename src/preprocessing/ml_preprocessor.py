import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import sys
import os
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging
from datetime import datetime
from src.models.ml_model import MLModel
from src.features.technical_indicators import TechnicalIndicators

# Ajout du chemin racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import explicite depuis le module
from src.preprocessing.data_processor import DataProcessor, GoldDataProcessor

class MLPreprocessor:
    def __init__(self, instrument='XAUUSD'):
        self.feature_scaler = MinMaxScaler()  # Pour les features techniques
        self.price_scaler = MinMaxScaler()    # Dédié aux prix OHLC
        self.scaler = MinMaxScaler()
        self.instrument = instrument
        self.indicator_calculator = TechnicalIndicators()  # Nouvelle instance
        
        # Initialisation du processeur approprié
        if 'XAU' in instrument:
            self.data_processor = GoldDataProcessor()
        else:
            self.data_processor = DataProcessor(instrument_type='forex')
        
        self.expected_price_ranges = {
            'XAUUSD': {
                'min': 1000,  # Prix minimum attendu pour l'or en USD
                'max': 3000,  # Prix maximum attendu pour l'or en USD
                'unit': 'USD/oz',
                'multiplier': 1  # Multiplier par défaut
            },
            # Ajouter d'autres instruments si nécessaire
        }
        
        # Configuration du logging
        self.logger = self._setup_logging()
        self.logger.info("Initialisation du MLPreprocessor")
        
    def _setup_logging(self):
        """Configure le système de logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        
        # Handler pour fichier
        os.makedirs('logs/preprocessing', exist_ok=True)
        fh = logging.FileHandler(
            f'logs/preprocessing/ml_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        
        # Handler pour console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def log_dataframe_stats(self, df, stage=""):
        """Log les statistiques d'un DataFrame"""
        self.logger.info(f"\nStatistiques DataFrame {stage}:")
        self.logger.info(f"Shape: {df.shape}")
        self.logger.info(f"Colonnes: {df.columns.tolist()}")
        self.logger.info(f"Types de données:\n{df.dtypes}")
        self.logger.info(f"Valeurs manquantes:\n{df.isnull().sum()}")
        
        # Statistiques numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            self.logger.info("\nStatistiques descriptives:")
            self.logger.info(f"\n{df[numeric_cols].describe()}")
    
    def prepare_features(self, df, instrument='XAUUSD'):
        try:
            df = df.copy()
            
            # Supprimer les doublons si présents
            if df.index.duplicated().any():
                self.logger.warning("Suppression des doublons dans l'index")
                df = df[~df.index.duplicated(keep='first')]
            
            # Traitement des données manquantes
            df = df.ffill().bfill()
            
            # Ajout des indicateurs techniques avec gestion d'erreur
            try:
                df = self.data_processor.add_technical_indicators(df)
            except Exception as e:
                self.logger.error(f"Erreur lors de l'ajout des indicateurs: {str(e)}")
                # Utiliser les indicateurs de base en fallback
                df = self._add_basic_indicators(df)
            
            # Vérification finale
            self.log_dataframe_stats(df, "final")
            
            return df, df[['Open', 'High', 'Low', 'Close']].copy()
            
        except Exception as e:
            self.logger.error(f"Erreur dans prepare_features: {str(e)}")
            raise

    def _add_basic_indicators(self, df):
        """Ajoute des indicateurs techniques de base en cas d'échec des indicateurs principaux"""
        try:
            # SMA basiques
            for period in [20, 50]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # RSI basique
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Volatilité
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            
            return df
        except Exception as e:
            self.logger.error(f"Erreur dans _add_basic_indicators: {str(e)}")
            return df  # Retourner le DataFrame original si même les indicateurs basiques échouent
    
    def calculate_atr(self, df, period=14):
        """Délègue le calcul de l'ATR à TechnicalIndicators"""
        return self.indicator_calculator.atr(
            df['High'], df['Low'], df['Close'], period
        )
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Délègue le calcul des bandes de Bollinger à TechnicalIndicators"""
        return self.indicator_calculator.bollinger_bands(
            df['Close'], period, std_dev
        )
    
    def prepare_target(self, df, horizon=5):
        """Prépare la variable cible"""
        df = df.copy()
        
        # Calculer la variation de prix future
        df['Target'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
        
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna(subset=['Target'])
        
        return df['Target']
    
    def create_sequences(self, data_dict, sequence_length=15, prediction_horizon=12, 
                       target_samples=3000, pip_threshold=0.0025):
        """Crée des séquences avec tailles alignées"""
        try:
            self.logger.info("\nDébut de la création des séquences")
            self.logger.info(f"Paramètres: sequence_length={sequence_length}, "
                           f"prediction_horizon={prediction_horizon}, "
                           f"target_samples={target_samples}")
            
            # Utiliser directement prepare_training_data sans récursion
            reference_tf = '5m'  # Timeframe de référence
            if reference_tf not in data_dict:
                reference_tf = list(data_dict.keys())[0]
                self.logger.warning(f"Timeframe 5m non trouvé, utilisation de {reference_tf}")
            
            sequences_dict = {}
            min_samples = float('inf')
            
            # 1. Première passe : créer les séquences pour chaque timeframe
            for tf, features in data_dict.items():
                try:
                    self.logger.info(f"\nTraitement du timeframe {tf}")
                    
                    if features.empty:
                        self.logger.warning(f"Pas de données pour {tf}, ignoré")
                        continue
                    
                    # Extraction et préparation
                    feature_cols = [col for col in features.columns if col not in ['Date']]
                    feature_data = features[feature_cols].values
                    
                    n_samples = len(features) - sequence_length - prediction_horizon
                    n_features = len(feature_cols)
                    
                    if n_samples <= 0:
                        self.logger.warning(f"Pas assez de données pour {tf}, ignoré")
                        continue
                    
                    # Création des séquences
                    X_sequences = np.zeros((n_samples, sequence_length, n_features))
                    
                    # Utiliser une approche vectorisée pour créer les séquences
                    for i in range(n_samples):
                        X_sequences[i] = feature_data[i:i+sequence_length]
                    
                    sequences_dict[tf] = {
                        'X': X_sequences
                    }
                    
                    min_samples = min(min_samples, len(X_sequences))
                    self.logger.info(f"Séquences créées: {X_sequences.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Erreur pour {tf}: {str(e)}")
                    continue
            
            if not sequences_dict:
                raise ValueError("Aucune séquence n'a pu être créée")
            
            # 2. Créer les labels une seule fois
            try:
                reference_data = data_dict[reference_tf]
                y_labels = self._create_labels(reference_data, prediction_horizon, pip_threshold)
                
                # Ajuster la taille des labels
                y_labels = y_labels[:min_samples]
                self.logger.info(f"Labels créés: {len(y_labels)}")
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la création des labels: {str(e)}")
                raise
            
            # 3. Aligner toutes les séquences sur la plus petite taille
            target_size = min(min_samples, target_samples)
            self.logger.info(f"\nAlignement des séquences sur {target_size} échantillons")
            
            aligned_sequences = {}
            for tf, sequences in sequences_dict.items():
                try:
                    X = sequences['X']
                    
                    if len(X) > target_size:
                        # Sous-échantillonnage aléatoire
                        indices = np.random.choice(len(X), target_size, replace=False)
                        X = X[indices]
                    
                    # Split train/test (80/20)
                    train_size = int(0.8 * target_size)
                    aligned_sequences[tf] = {
                        'X': X[:train_size],
                        'X_test': X[train_size:],
                    }
                    
                    self.logger.info(f"\n{tf}:")
                    self.logger.info(f"• Shape finale: {X[:train_size].shape}")
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'alignement de {tf}: {str(e)}")
                    continue
            
            if not aligned_sequences:
                raise ValueError("Échec de l'alignement des séquences")
            
            # Ajuster les labels également
            y_train = y_labels[:train_size]
            y_test = y_labels[train_size:target_size]
            
            # Vérification finale
            train_sizes = [seq['X'].shape[0] for seq in aligned_sequences.values()]
            if len(set(train_sizes)) != 1:
                raise ValueError(f"Tailles incohérentes après alignement: {train_sizes}")
            
            # Modifier la structure de retour pour être compatible avec l'entraînement
            result = {
                'train': {
                    'X': {tf: seq['X'] for tf, seq in aligned_sequences.items()},
                    'y': y_train
                },
                'test': {
                    'X': {tf: seq['X_test'] for tf, seq in aligned_sequences.items()},
                    'y': y_test
                }
            }
            
            # Log des shapes finales
            self.logger.info("\nShapes finales:")
            self.logger.info("Train:")
            for tf, X in result['train']['X'].items():
                self.logger.info(f"• {tf}: {X.shape}")
            self.logger.info(f"• y: {result['train']['y'].shape}")
            
            self.logger.info("\nTest:")
            for tf, X in result['test']['X'].items():
                self.logger.info(f"• {tf}: {X.shape}")
            self.logger.info(f"• y: {result['test']['y'].shape}")
            
            # Vérification de la cohérence des données
            train_lengths = [X.shape[0] for X in result['train']['X'].values()]
            test_lengths = [X.shape[0] for X in result['test']['X'].values()]
            
            if len(set(train_lengths)) != 1:
                raise ValueError(f"Tailles incohérentes dans l'ensemble d'entraînement: {train_lengths}")
            if len(set(test_lengths)) != 1:
                raise ValueError(f"Tailles incohérentes dans l'ensemble de test: {test_lengths}")
            if train_lengths[0] != len(result['train']['y']):
                raise ValueError("Nombre d'échantillons X et y incohérent pour l'entraînement")
            if test_lengths[0] != len(result['test']['y']):
                raise ValueError("Nombre d'échantillons X et y incohérent pour le test")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur dans create_sequences: {str(e)}")
            raise

    def _create_labels(self, df, horizon, pip_threshold):
        """Crée les labels sans récursion"""
        future_returns = df['Close'].pct_change(periods=horizon).shift(-horizon)
        
        # Initialiser les labels
        labels = np.zeros(len(df) - horizon)
        
        # Classifier selon les seuils
        returns = future_returns[:-horizon].values  # Éviter les NaN à la fin
        labels[returns > pip_threshold] = 2  # Hausse
        labels[returns < -pip_threshold] = 0  # Baisse
        labels[(returns >= -pip_threshold) & (returns <= pip_threshold)] = 1  # Neutre
        
        return labels
    
    def _balance_classes_vectorized(self, X, y, timeframe=None):
        """Version vectorisée optimisée de l'équilibrage des classes"""
        try:
            self.logger.info("\nÉquilibrage vectorisé des classes:")
            
            # Conversion en arrays numpy pour optimisation
            X = np.asarray(X)
            y = np.asarray(y)
            
            # Analyse initiale des classes
            unique_classes, counts = np.unique(y, return_counts=True)
            target_size = np.max(counts)
            class_stats = dict(zip(unique_classes, counts))
            
            self.logger.info("Distribution initiale:")
            for cls, count in class_stats.items():
                self.logger.info(f"• Classe {cls}: {count} échantillons")
            
            # Préallocation des arrays finaux
            total_samples = target_size * len(unique_classes)
            X_shape = (total_samples,) + X.shape[1:]
            X_balanced = np.zeros(X_shape, dtype=X.dtype)
            y_balanced = np.zeros(total_samples, dtype=y.dtype)
            
            # Traitement vectorisé par classe
            current_idx = 0
            for label in unique_classes:
                try:
                    # Sélection vectorisée des échantillons de la classe
                    mask = (y == label)
                    X_class = X[mask]
                    n_samples = len(X_class)
                    
                    if n_samples == 0:
                        continue
                    
                    # Calcul du nombre d'échantillons à générer
                    n_to_generate = target_size - n_samples
                    
                    if n_to_generate <= 0:
                        # Sous-échantillonnage si nécessaire
                        indices = np.random.choice(n_samples, target_size, replace=False)
                        samples = X_class[indices]
                    else:
                        # Sur-échantillonnage vectorisé
                        samples = np.zeros((target_size,) + X_class.shape[1:], dtype=X_class.dtype)
                        samples[:n_samples] = X_class
                        
                        # Paramètres d'augmentation selon le timeframe et la classe
                        aug_params = self._get_augmentation_params(timeframe, label)
                        
                        # Génération vectorisée des nouveaux échantillons
                        new_samples = self._generate_samples_vectorized(
                            X_class,
                            n_to_generate,
                            aug_params
                        )
                        
                        samples[n_samples:] = new_samples
                    
                    # Assignation vectorisée au résultat final
                    end_idx = current_idx + target_size
                    X_balanced[current_idx:end_idx] = samples
                    y_balanced[current_idx:end_idx] = label
                    current_idx = end_idx
                    
                except Exception as e:
                    self.logger.error(f"Erreur pour classe {label}: {str(e)}")
                    continue
            
            # Mélange final vectorisé
            shuffle_indices = np.random.permutation(current_idx)
            X_balanced = X_balanced[shuffle_indices]
            y_balanced = y_balanced[shuffle_indices]
            
            # Vérification finale
            final_counts = np.bincount(y_balanced.astype(int))
            self.logger.info("\nDistribution finale:")
            for label, count in enumerate(final_counts):
                self.logger.info(f"• Classe {label}: {count} échantillons")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.error(f"Erreur dans _balance_classes_vectorized: {str(e)}")
            raise

    def _get_augmentation_params(self, timeframe, label):
        """Retourne les paramètres d'augmentation optimisés selon le contexte"""
        base_params = {
            'noise_scale': 0.01,
            'scale_range': (0.98, 1.02),
            'mix_ratio_alpha': 2.0
        }
        
        # Ajustements selon le timeframe
        if timeframe == '1M':
            base_params.update({
                'noise_scale': 0.03,
                'scale_range': (0.97, 1.03),
                'mix_ratio_alpha': 3.0
            })
        elif timeframe == '1w':
            base_params.update({
                'noise_scale': 0.02,
                'scale_range': (0.97, 1.03),
                'mix_ratio_alpha': 2.5
            })
        
        # Ajustements selon la classe
        if label == 1:  # Classe neutre
            base_params['noise_scale'] *= 0.8  # Réduire le bruit
            base_params['mix_ratio_alpha'] *= 1.2  # Plus de mélange
        
        return base_params

    def _generate_samples_vectorized(self, X_base, n_samples, params):
        """Génère de nouveaux échantillons de manière vectorisée"""
        try:
            n_base = len(X_base)
            if n_base < 2:
                return self._augment_single_sample_vectorized(X_base[0], n_samples, params)
            
            # Sélection vectorisée des paires
            idx1 = np.random.randint(0, n_base, n_samples)
            idx2 = np.random.randint(0, n_base, n_samples)
            
            # Éviter les paires identiques
            mask = idx1 == idx2
            idx2[mask] = (idx2[mask] + 1) % n_base
            
            # Création des ratios de mélange
            mix_ratios = np.random.beta(
                params['mix_ratio_alpha'],
                params['mix_ratio_alpha'],
                size=(n_samples,) + (1,) * (X_base.ndim - 1)
            )
            
            # Interpolation vectorisée
            samples = (
                X_base[idx1] * mix_ratios +
                X_base[idx2] * (1 - mix_ratios)
            )
            
            # Application vectorisée des transformations
            samples = self._apply_transformations_vectorized(samples, params)
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Erreur dans _generate_samples_vectorized: {str(e)}")
            raise

    def _apply_transformations_vectorized(self, samples, params):
        """Applique les transformations de manière vectorisée"""
        try:
            # Mise à l'échelle vectorisée
            scale_factors = np.random.uniform(
                params['scale_range'][0],
                params['scale_range'][1],
                size=(len(samples),) + (1,) * (samples.ndim - 1)
            )
            samples = samples * scale_factors
            
            # Bruit gaussien vectorisé
            noise = np.random.normal(
                0,
                params['noise_scale'],
                samples.shape
            )
            samples = samples + noise
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Erreur dans _apply_transformations_vectorized: {str(e)}")
            raise

    def _augment_single_sample_vectorized(self, sample, n_samples, params):
        """Augmentation vectorisée pour un seul échantillon"""
        try:
            # Répétition vectorisée
            samples = np.repeat(sample[np.newaxis], n_samples, axis=0)
            
            # Application des transformations
            return self._apply_transformations_vectorized(samples, params)
            
        except Exception as e:
            self.logger.error(f"Erreur dans _augment_single_sample_vectorized: {str(e)}")
            raise

    def prepare_multi_timeframe_features(self, data_dict):
        """Prépare les features pour chaque timeframe"""
        processed_data = {}
        price_data_all = []  # Pour collecter les données de prix de tous les timeframes
        
        # Première passe : collecter toutes les données de prix pour le fit du scaler
        for timeframe, df in data_dict.items():
            df = df.copy()
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
            except Exception as e:
                self.logger.warning(f"Erreur format date standard, tentative format alternatif: {str(e)}")
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    self.logger.error(f"Échec conversion dates: {str(e)}")
                    continue
            
            df.set_index('Date', inplace=True)
            price_cols = ['Open', 'High', 'Low', 'Close']
            price_data = df[price_cols].copy()
            price_data_all.append(price_data)
        
        # Fit du price_scaler sur toutes les données de prix combinées
        if price_data_all:
            combined_prices = pd.concat(price_data_all)
            self.price_scaler.fit(combined_prices)
            self.logger.info("\nStatistiques de normalisation des prix:")
            self.logger.info(f"Min: {self.price_scaler.data_min_}")
            self.logger.info(f"Max: {self.price_scaler.data_max_}")
        
        # Deuxième passe : traitement complet des données
        for timeframe, df in data_dict.items():
            self.logger.info(f"\nTraitement du timeframe {timeframe}")
            df = df.copy()
            
            # Vérification et conversion des dates déjà effectuées
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            
            # Debug des prix avant traitement
            self.logger.info("\nPrix avant normalisation:")
            self.logger.info(df[['Open', 'High', 'Low', 'Close']].describe())
            
            # Séparer prix et autres features
            price_cols = ['Open', 'High', 'Low', 'Close']
            price_data = df[price_cols].copy()
            other_features = df.drop(price_cols + ['Volume'], axis=1, errors='ignore')
            
            # Calculer les returns et la volatilité sur les prix originaux
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=50).std()
            
            # Normaliser les prix
            try:
                normalized_prices = pd.DataFrame(
                    self.price_scaler.transform(price_data),
                    columns=price_cols,
                    index=price_data.index
                )
                self.logger.info("\nPrix après normalisation:")
                self.logger.info(normalized_prices.describe())
            except Exception as e:
                self.logger.error(f"Erreur normalisation prix: {str(e)}")
                continue
            
            # Calculer les features techniques sur les prix normalisés
            try:
                technical_features = self.indicator_calculator.calculate_all(
                    pd.concat([normalized_prices, df[['Volume']]], axis=1)
                )
                self.logger.info("\nIndicateurs calculés:")
                self.logger.info(f"• Nombre d'indicateurs: {len(technical_features.columns)}")
            except Exception as e:
                self.logger.error(f"Erreur calcul indicateurs: {str(e)}")
                technical_features = pd.DataFrame(index=normalized_prices.index)
            
            # Normaliser les features techniques
            try:
                # Exclure les colonnes déjà normalisées
                tech_cols = [col for col in technical_features.columns 
                           if col not in price_cols + ['Volume', 'Returns']]
                
                if tech_cols:
                    tech_features = technical_features[tech_cols]
                    # Remplacer les infinis et gérer les NaN
                    tech_features = tech_features.replace([np.inf, -np.inf], np.nan)
                    tech_features = tech_features.ffill().bfill()
                    
                    # Normaliser
                    normalized_features = pd.DataFrame(
                        self.feature_scaler.fit_transform(tech_features),
                        columns=tech_cols,
                        index=tech_features.index
                    )
                    
                    self.logger.info("\nFeatures techniques après normalisation:")
                    self.logger.info(normalized_features.describe())
                else:
                    normalized_features = pd.DataFrame(index=normalized_prices.index)
            except Exception as e:
                self.logger.error(f"Erreur normalisation features: {str(e)}")
                normalized_features = pd.DataFrame(index=normalized_prices.index)
            
            # Combiner toutes les features
            final_features = pd.concat(
                [normalized_prices, normalized_features],
                axis=1
            )
            
            # Vérifications finales
            self.logger.info("\nVérifications finales:")
            self.logger.info(f"Shape: {final_features.shape}")
            self.logger.info(f"NaN count: {final_features.isna().sum().sum()}")
            self.logger.info(f"Inf count: {np.isinf(final_features.values).sum()}")
            
            processed_data[timeframe] = final_features
        
        return processed_data

    def inverse_transform_prices(self, normalized_prices):
        """Convertit les prix normalisés en prix réels"""
        if isinstance(normalized_prices, pd.DataFrame):
            return pd.DataFrame(
                self.price_scaler.inverse_transform(normalized_prices),
                columns=normalized_prices.columns,
                index=normalized_prices.index
            )
        return self.price_scaler.inverse_transform(normalized_prices)

    def create_multi_timeframe_sequences(self, data_dict, y, sequence_length=10):
        """Crée des séquences pour chaque timeframe"""
        sequences = {}
        
        # Convertir y en array numpy si ce n'est pas déjà fait
        if isinstance(y, pd.Series):
            y = y.values
        
        # Définir les multiplicateurs pour chaque timeframe
        tf_multipliers = {
            '5m': 288,    # 24h * 12 (12 périodes de 5min par heure)
            '15m': 96,    # 24h * 4
            '30m': 48,    # 24h * 2
            '1h': 24,     # 24h
            '4h': 6,      # 24h / 4
            '1d': 1,      # référence
            '1w': 1/7,    # environ 0.14
            '1M': 1/30    # environ 0.033
        }
        
        for timeframe, features in data_dict.items():
            self.logger.info(f"Création des séquences pour {timeframe}")
            multiplier = tf_multipliers[timeframe]
            adj_sequence_length = int(sequence_length * multiplier)
            
            # Convertir features en array numpy si ce n'est pas déjà fait
            if isinstance(features, pd.DataFrame):
                features = features.values
            
            X_seq, y_seq = [], []
            
            # S'assurer que nous ne dépassons pas les limites
            max_idx = min(len(features), len(y)) - adj_sequence_length
            
            for i in range(max_idx):
                if i + adj_sequence_length < len(features):
                    X_seq.append(features[i:i+adj_sequence_length])
                    y_seq.append(y[i+adj_sequence_length-1])  # -1 pour éviter le dépassement
            
            if X_seq:  # Vérifier qu'il y a des séquences
                sequences[timeframe] = (np.array(X_seq), np.array(y_seq))
                self.logger.info(f"Séquences {timeframe}: {len(X_seq)} échantillons")
            else:
                self.logger.warning(f"Attention: Aucune séquence créée pour {timeframe}")
        
        return sequences
    
    def add_technical_indicators(self, df):
        """Délègue le calcul des indicateurs à TechnicalIndicators"""
        return self.indicator_calculator.calculate_all(df)
    
    def prepare_data_for_training(self, 
                                data_dict: Dict[str, pd.DataFrame],
                                sequence_length: int = 15,
                                prediction_horizon: int = 12,
                                train_size: float = 0.8,
                                target_samples: int = 3000) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement du modèle ML"""
        
        # Préparation des données avec les nouveaux paramètres
        X_train, X_test, y_train, y_test = self.data_processor.prepare_training_data(
            data_dict,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            train_size=train_size
        )
        
        # Vérification des distributions
        self.logger.info("\nDistribution finale des données:")
        self.logger.info("\nTrain set:")
        self._print_distribution(y_train)
        self.logger.info("\nTest set:")
        self._print_distribution(y_test)
        
        return X_train, y_train, X_test, y_test

    def _print_distribution(self, y):
        """Affiche la distribution détaillée des labels"""
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        total = sum(counts)
        
        self.logger.info("Distribution des classes:")
        for label, count in sorted(dist.items()):
            percentage = (count/total) * 100
            self.logger.info(f"Classe {label:2}: {count:4d} échantillons ({percentage:5.2f}%)")
        
        # Calculer des statistiques supplémentaires
        if len(unique) > 1:
            max_to_min_ratio = max(counts) / min(counts)
            self.logger.info(f"\nRatio max/min: {max_to_min_ratio:.2f}")
            self.logger.info(f"Écart-type des proportions: {np.std(counts/total):.3f}")

    def check_data_balance(self, y_train, y_test):
        """Vérifie l'équilibre des données train/test"""
        self.logger.info("\nVérification de l'équilibre des données:")
        
        # Distribution train
        self.logger.info("\nEnsemble d'entraînement:")
        self._print_distribution(y_train)
        
        # Distribution test
        self.logger.info("\nEnsemble de test:")
        self._print_distribution(y_test)
        
        # Vérifier les ratios
        train_dist = np.bincount(y_train.astype(int) + 1) / len(y_train)
        test_dist = np.bincount(y_test.astype(int) + 1) / len(y_test)
        
        self.logger.info("\nDifférence de distribution train/test:")
        for i, (train_pct, test_pct) in enumerate(zip(train_dist, test_dist)):
            diff = abs(train_pct - test_pct) * 100
            self.logger.info(f"Classe {i-1}: {diff:.2f}% de différence")
        
        return np.all(abs(train_dist - test_dist) < 0.1)  # Retourne True si bien équilibré

    def calculate_rsi(self, df, period=14):
        """Calcule le RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, df):
        """Calcule le MACD"""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def add_gaussian_noise(self, X, scale=0.01):
        """
        Ajoute du bruit gaussien aux données existantes
        
        Args:
            X: Features (2D array)
            scale: Échelle du bruit gaussien
            
        Returns:
            X avec bruit gaussien ajouté
        """
        noise = np.random.normal(0, scale, X.shape)
        return X + noise

    def interpolate_neutral_samples(self, X, y, timeframe=None):
        """
        Crée des échantillons neutres supplémentaires par interpolation avec gestion robuste
        """
        try:
            neutral_mask = y == 0
            neutral_indices = np.where(neutral_mask)[0]
            n_neutral = len(neutral_indices)
            
            self.logger.info(f"\nDébut interpolation pour {timeframe}:")
            self.logger.info(f"Échantillons neutres disponibles: {n_neutral}")
            
            # Vérification du nombre minimal d'échantillons
            if n_neutral < 2:
                self.logger.warning("Pas assez d'échantillons neutres pour interpoler")
                if n_neutral == 1:
                    # Cas spécial : un seul échantillon neutre
                    self.logger.info("Utilisation de l'augmentation de données simple")
                    return self._augment_single_sample(X[neutral_indices[0]], n_samples=5)
                return X, y
            
            # Ajustements spéciaux selon le timeframe
            if timeframe == '1M':
                target_ratio = 0.11  # Ratio cible pour 1M
                current_ratio = n_neutral / len(y)
                n_samples = int((target_ratio * len(y) - n_neutral) / (1 - target_ratio))
                n_samples = max(n_samples, 5)  # Au moins 5 nouveaux échantillons
            else:
                n_samples = min(n_neutral * 2, 100)  # Limite raisonnable
            
            self.logger.info(f"Génération de {n_samples} nouveaux échantillons")
            
            # Création des paires avec vérification
            try:
                pairs = self._create_diverse_pairs(X[neutral_indices], n_samples)
                if pairs is None:
                    return X, y
                idx1, idx2 = pairs
                
                # Interpolation avec bruit adaptatif
                X_interp = self._interpolate_with_noise(
                    X[neutral_indices][idx1],
                    X[neutral_indices][idx2],
                    timeframe
                )
                
                # Vérification de la qualité des échantillons générés
                if not self._validate_samples(X_interp):
                    self.logger.warning("Échantillons générés invalides, retour aux données originales")
                    return X, y
                
                # Combinaison des données
                X_combined = np.vstack([X, X_interp])
                y_combined = np.concatenate([y, np.zeros(len(X_interp))])
                
                return X_combined, y_combined
                
            except Exception as e:
                self.logger.error(f"Erreur lors de l'interpolation: {str(e)}")
                return X, y
            
        except Exception as e:
            self.logger.error(f"Erreur globale dans interpolate_neutral_samples: {str(e)}")
            return X, y

    def _create_diverse_pairs(self, X, n_samples):
        """Crée des paires diversifiées pour l'interpolation"""
        try:
            if len(X) < 2:
                return None
            
            # Utiliser KNN pour trouver des paires similaires
            n_neighbors = min(5, len(X))
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(X.reshape(len(X), -1))
            distances, indices = knn.kneighbors()
            
            # Sélection des paires
            idx1 = []
            idx2 = []
            for _ in range(n_samples):
                base_idx = np.random.randint(len(X))
                neighbor_idx = indices[base_idx][np.random.randint(1, n_neighbors)]
                idx1.append(base_idx)
                idx2.append(neighbor_idx)
            
            return np.array(idx1), np.array(idx2)
            
        except Exception as e:
            self.logger.error(f"Erreur dans create_diverse_pairs: {str(e)}")
            return None

    def _interpolate_with_noise(self, X1, X2, timeframe):
        """Interpolation avec bruit adaptatif selon le timeframe"""
        try:
            # Paramètres de bruit selon le timeframe
            noise_params = {
                '1M': {'scale': 0.03, 'ratio_alpha': 3},
                '1w': {'scale': 0.02, 'ratio_alpha': 2},
                '1d': {'scale': 0.015, 'ratio_alpha': 1.5},
                'default': {'scale': 0.01, 'ratio_alpha': 1}
            }
            
            params = noise_params.get(timeframe, noise_params['default'])
            
            # Ratios d'interpolation avec distribution beta
            ratios = np.random.beta(params['ratio_alpha'], params['ratio_alpha'], 
                                  size=(len(X1), 1, 1))
            
            # Interpolation
            interpolated = X1 * ratios + X2 * (1 - ratios)
            
            # Bruit adaptatif
            noise = np.random.normal(0, params['scale'], interpolated.shape)
            interpolated += noise
            
            return interpolated
            
        except Exception as e:
            self.logger.error(f"Erreur dans interpolate_with_noise: {str(e)}")
            raise

    def _validate_samples(self, X):
        """Valide la qualité des échantillons générés"""
        try:
            # Vérification des valeurs aberrantes
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                return False
            
            # Vérification de la plage de valeurs
            if np.any(X > 10) or np.any(X < -10):  # Valeurs normalisées attendues
                return False
            
            # Vérification de la variance
            if np.var(X) < 1e-6:  # Trop peu de variance
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur dans validate_samples: {str(e)}")
            return False

    def _augment_single_sample(self, sample, n_samples):
        """Augmentation de données pour un seul échantillon"""
        try:
            # Créer plusieurs versions avec bruit
            noise_scales = np.linspace(0.01, 0.03, n_samples)
            augmented = []
            
            for scale in noise_scales:
                noise = np.random.normal(0, scale, sample.shape)
                augmented.append(sample + noise)
            
            return np.array(augmented)
            
        except Exception as e:
            self.logger.error(f"Erreur dans augment_single_sample: {str(e)}")
            return np.array([sample])

    def hybrid_balance(self, X, y, target_counts, timeframe=None):
        """Méthode hybride robuste pour l'équilibrage des données"""
        try:
            self.logger.info(f"\nDébut équilibrage hybride pour {timeframe}")
            current_counts = {label: np.sum(y == label) for label in np.unique(y)}
            
            # Vérification des données minimales
            min_samples_required = 2
            for label, count in current_counts.items():
                if count < min_samples_required:
                    self.logger.warning(f"Trop peu d'échantillons pour classe {label} ({count})")
                    if count == 1:
                        # Augmenter artificiellement les échantillons uniques
                        X, y = self._handle_single_sample_class(X, y, label)
                    elif count == 0:
                        # Ajuster les objectifs pour ignorer cette classe
                        target_counts.pop(label, None)
            
            X_res, y_res = X.copy(), y.copy()
            
            for label in target_counts:
                try:
                    if label not in current_counts:
                        continue
                    
                    current = current_counts[label]
                    target = target_counts[label]
                    
                    if current >= target:
                        continue
                    
                    # Nombre d'échantillons à générer
                    n_needed = target - current
                    self.logger.info(f"Génération de {n_needed} échantillons pour classe {label}")
                    
                    # Stratégie adaptative selon le nombre d'échantillons disponibles
                    if current < 5:
                        X_new, y_new = self._conservative_augmentation(
                            X[y == label], label, n_needed
                        )
                    else:
                        X_new, y_new = self._standard_augmentation(
                            X[y == label], label, n_needed, timeframe
                        )
                    
                    if X_new is not None:
                        X_res = np.vstack([X_res, X_new])
                        y_res = np.concatenate([y_res, y_new])
                    
                except Exception as e:
                    self.logger.error(f"Erreur pour classe {label}: {str(e)}")
                    continue
            
            return X_res, y_res
            
        except Exception as e:
            self.logger.error(f"Erreur dans hybrid_balance: {str(e)}")
            return X, y

    def _handle_single_sample_class(self, X, y, label):
        """Gère le cas d'une classe avec un seul échantillon"""
        try:
            mask = y == label
            single_sample = X[mask][0]
            augmented = self._augment_single_sample(single_sample, n_samples=5)
            
            X_new = np.vstack([X, augmented])
            y_new = np.concatenate([y, np.full(len(augmented), label)])
            
            return X_new, y_new
            
        except Exception as e:
            self.logger.error(f"Erreur dans handle_single_sample_class: {str(e)}")
            return X, y

    def _conservative_augmentation(self, X, label, n_needed):
        """Augmentation conservative pour peu d'échantillons"""
        try:
            # Utiliser uniquement du bruit et des transformations simples
            augmented = []
            n_base = len(X)
            
            for i in range(n_needed):
                base_idx = i % n_base
                sample = X[base_idx].copy()
                
                # Bruit minimal
                noise = np.random.normal(0, 0.01, sample.shape)
                # Légère mise à l'échelle
                scale = np.random.uniform(0.98, 1.02)
                
                augmented.append(sample * scale + noise)
            
            return np.array(augmented), np.full(n_needed, label)
            
        except Exception as e:
            self.logger.error(f"Erreur dans conservative_augmentation: {str(e)}")
            return None, None

    def _standard_augmentation(self, X, label, n_needed, timeframe):
        """Augmentation standard avec plus d'échantillons disponibles"""
        try:
            # Combiner interpolation et augmentation
            n_interp = int(n_needed * 0.6)
            n_augment = n_needed - n_interp
            
            # Interpolation
            X_interp, _ = self.interpolate_neutral_samples(X, np.zeros(len(X)), timeframe)
            if len(X_interp) > len(X):
                X_interp = X_interp[len(X):]  # Garder uniquement les nouveaux échantillons
            else:
                X_interp = np.empty((0, *X.shape[1:]))
            
            # Augmentation
            X_aug = self.augment_samples(X, n_augment)
            
            # Combiner
            X_new = np.vstack([X_interp, X_aug]) if len(X_interp) > 0 else X_aug
            y_new = np.full(len(X_new), label)
            
            return X_new, y_new
            
        except Exception as e:
            self.logger.error(f"Erreur dans standard_augmentation: {str(e)}")
            return None, None

    def augment_samples(self, X, scale_range=(0.98, 1.02), noise_scale=0.01):
        """
        Applique diverses techniques d'augmentation de données
        
        Args:
            X: Features à augmenter
            scale_range: Plage de mise à l'échelle
            noise_scale: Échelle du bruit gaussien
            
        Returns:
            X augmenté
        """
        # Scaling aléatoire
        scale = np.random.uniform(*scale_range, size=(len(X), 1))
        X_scaled = X * scale
        
        # Ajout de bruit gaussien
        noise = np.random.normal(0, noise_scale, X.shape)
        X_noisy = X_scaled + noise
        
        return X_noisy

    def optimize_hyperparameters(self, train_data, n_combinations=20, n_folds=5):
        """Optimisation avec validation des scores"""
        try:
            self.logger.info("\nDébut de l'optimisation des hyperparamètres")
            
            # Extraire et standardiser les données
            X_dict = train_data['X']  # {'5m': data, '15m': data, ...}
            y = train_data['y']
            
            # Standardiser les clés des données
            standardized_X = {}
            for tf, data in X_dict.items():
                input_key = f'input_{tf}'  # Convertir '5m' en 'input_5m'
                standardized_X[input_key] = data
            
            timeframes = list(X_dict.keys())
            self.logger.info("\nTimeframes disponibles:")
            for tf in timeframes:
                self.logger.info(f"• {tf}")
            
            # Vérifier la cohérence des shapes
            reference_shape = None
            for tf, X in standardized_X.items():
                if reference_shape is None:
                    reference_shape = X.shape[0]
                elif X.shape[0] != reference_shape:
                    raise ValueError(
                        f"Tailles incohérentes: {tf}={X.shape[0]} != {reference_shape}"
                    )
            
            self.logger.info("\nConfiguration des données:")
            for tf, X in standardized_X.items():
                self.logger.info(f"• {tf}: shape={X.shape}")
            self.logger.info(f"• Labels: shape={y.shape}")
            
            # Générer les combinaisons d'hyperparamètres
            param_combinations = self._generate_param_combinations(n_combinations)
            
            best_score = float('inf')
            best_params = None
            
            # Évaluation de chaque combinaison
            for i, params in enumerate(param_combinations, 1):
                self.logger.info(f"\nTest combinaison {i}/{n_combinations}")
                self.logger.info(f"Paramètres: {params}")
                
                fold_scores = []
                
                for fold in range(n_folds):
                    try:
                        # Indices pour la validation croisée
                        train_idx = np.arange(len(y))
                        np.random.shuffle(train_idx)
                        split = int(0.8 * len(train_idx))
                        fold_train_idx = train_idx[:split]
                        fold_val_idx = train_idx[split:]
                        
                        # Créer les sous-ensembles pour chaque timeframe
                        X_train_fold = {
                            tf: X[fold_train_idx] for tf, X in standardized_X.items()
                        }
                        X_val_fold = {
                            tf: X[fold_val_idx] for tf, X in standardized_X.items()
                        }
                        y_train_fold = y[fold_train_idx]
                        y_val_fold = y[fold_val_idx]
                        
                        # Entraînement et évaluation
                        reference_tf = f"input_{timeframes[0]}"  # ex: 'input_5m'
                        input_shape = (
                            standardized_X[reference_tf].shape[1],
                            standardized_X[reference_tf].shape[2]
                        )
                        
                        model = MLModel(
                            input_shape=input_shape,
                            n_classes=3,
                            learning_rate=params['learning_rate'],
                            lstm_units=params['lstm_units'],
                            dense_units=params['dense_units'],
                            dropout_rate=params['dropout_rate']
                        )
                        
                        # Entraînement avec les données standardisées
                        history = model.fit(
                            X_train_fold,
                            y_train_fold,
                            validation_data=(X_val_fold, y_val_fold),
                            batch_size=params['batch_size'],
                            epochs=10,
                            verbose=0
                        )
                        
                        # Validation du score
                        val_loss = history.history['val_loss'][-1]
                        if np.isfinite(val_loss) and val_loss > 0:
                            fold_scores.append(val_loss)
                            self.logger.info(f"Fold {fold + 1}: score valide = {val_loss:.4f}")
                        else:
                            self.logger.warning(f"Score invalide ignoré: {val_loss}")
                        
                    except Exception as e:
                        self.logger.error(f"Erreur dans le fold {fold + 1}: {str(e)}")
                        continue
                
                # Calculer le score moyen uniquement sur les scores valides
                if fold_scores:
                    mean_score = np.mean(fold_scores)
                    std_score = np.std(fold_scores)
                    self.logger.info(f"Score moyen valide: {mean_score:.4f} (±{std_score:.4f})")
                    
                    if mean_score < best_score:
                        best_score = mean_score
                        best_params = params
                        self.logger.info("→ Nouveaux meilleurs paramètres")
                else:
                    self.logger.warning("Aucun score valide pour cette combinaison")
            
            if best_params is None:
                self.logger.warning("Utilisation des paramètres par défaut")
                best_params = self._get_default_params()
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Erreur dans l'optimisation: {str(e)}")
            return self._get_default_params()

    def _get_default_params(self):
        """Retourne les paramètres par défaut"""
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'dropout_rate': 0.3,
            'lstm_units': 64,
            'dense_units': 32
        }

    def _generate_param_combinations(self, n_combinations):
        """Génère des combinaisons d'hyperparamètres à tester"""
        param_space = {
            'learning_rate': np.logspace(-4, -2, 10),
            'batch_size': [32, 64, 128],
            'dropout_rate': np.linspace(0.1, 0.5, 5),
            'lstm_units': [32, 64, 128],
            'dense_units': [16, 32, 64]
        }
        
        combinations = []
        for _ in range(n_combinations):
            params = {
                'learning_rate': np.random.choice(param_space['learning_rate']),
                'batch_size': np.random.choice(param_space['batch_size']),
                'dropout_rate': np.random.choice(param_space['dropout_rate']),
                'lstm_units': np.random.choice(param_space['lstm_units']),
                'dense_units': np.random.choice(param_space['dense_units'])
            }
            combinations.append(params)
        
        return combinations

    def _create_model(self, input_shapes, **params):
        """Crée un modèle avec les hyperparamètres spécifiés"""
        # À implémenter selon l'architecture du modèle
        pass

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestion des outliers avec préservation de l'index"""
        try:
            df = df.copy()
            price_cols = ['Open', 'High', 'Low', 'Close']
            
            for col in df.columns:
                if col not in price_cols + ['Date', 'Volume']:
                    # Calcul des z-scores
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    
                    if outliers.any():
                        n_outliers = outliers.sum()
                        self.logger.info(f"Correction de {n_outliers} outliers dans {col}")
                        
                        # Winsorisation avec préservation de l'index
                        lower_bound = df[col].quantile(0.01)
                        upper_bound = df[col].quantile(0.99)
                        
                        df.loc[outliers, col] = df.loc[outliers, col].clip(
                            lower=lower_bound,
                            upper=upper_bound
                        )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _handle_outliers: {str(e)}")
            return df

    def prepare_data(self, data):
        # Renommer les clés pour correspondre au format attendu
        renamed_data = {
            f'input_{timeframe}': data[timeframe] 
            for timeframe in ['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        }
        return renamed_data

    def update_sequences(self, current_sequences, new_data_dict, max_sequences=1000):
        """
        Met à jour les séquences avec de nouvelles données en temps réel
        
        Args:
            current_sequences: Dict contenant les séquences actuelles par timeframe
            new_data_dict: Dict contenant les nouvelles données par timeframe
            max_sequences: Nombre maximum de séquences à conserver
        
        Returns:
            Dict avec les séquences mises à jour
        """
        try:
            self.logger.info("\nMise à jour des séquences en temps réel")
            updated_sequences = {}
            
            for timeframe, new_data in new_data_dict.items():
                try:
                    self.logger.info(f"\nTraitement du timeframe {timeframe}")
                    
                    # Vérifier si nous avons des séquences existantes pour ce timeframe
                    current_X = current_sequences.get(timeframe, {}).get('X', None)
                    current_y = current_sequences.get(timeframe, {}).get('y', None)
                    
                    # Préparation des nouvelles données
                    new_features = self._prepare_live_features(new_data, timeframe)
                    new_labels = self._create_live_labels(new_data)
                    
                    if current_X is None:
                        # Première initialisation
                        updated_sequences[timeframe] = {
                            'X': new_features,
                            'y': new_labels
                        }
                        continue
                    
                    # Concaténation des données
                    X_updated = np.concatenate([current_X, new_features], axis=0)
                    y_updated = np.concatenate([current_y, new_labels], axis=0)
                    
                    # Limitation du nombre de séquences
                    if len(X_updated) > max_sequences:
                        X_updated = X_updated[-max_sequences:]
                        y_updated = y_updated[-max_sequences:]
                    
                    # Vérifications de qualité
                    if self._validate_updated_sequences(X_updated, y_updated):
                        updated_sequences[timeframe] = {
                            'X': X_updated,
                            'y': y_updated
                        }
                        self.logger.info(f"✓ Séquences {timeframe} mises à jour:")
                        self.logger.info(f"• Shape X: {X_updated.shape}")
                        self.logger.info(f"• Shape y: {y_updated.shape}")
                    else:
                        self.logger.warning(f"⚠️ Validation échouée pour {timeframe}, conservation des anciennes séquences")
                        updated_sequences[timeframe] = current_sequences[timeframe]
                
                except Exception as e:
                    self.logger.error(f"Erreur pour {timeframe}: {str(e)}")
                    # Conserver les anciennes séquences en cas d'erreur
                    if timeframe in current_sequences:
                        updated_sequences[timeframe] = current_sequences[timeframe]
            
            return updated_sequences
            
        except Exception as e:
            self.logger.error(f"Erreur dans update_sequences: {str(e)}")
            return current_sequences

    def _prepare_live_features(self, new_data, timeframe):
        """Prépare les features pour les nouvelles données en temps réel"""
        try:
            # Normalisation des prix
            price_cols = ['Open', 'High', 'Low', 'Close']
            normalized_prices = self.price_scaler.transform(new_data[price_cols])
            
            # Calcul des indicateurs techniques
            df_normalized = pd.DataFrame(
                normalized_prices,
                columns=price_cols,
                index=new_data.index
            )
            df_normalized['Volume'] = new_data['Volume']
            
            technical_features = self.indicator_calculator.calculate_all(df_normalized)
            
            # Normalisation des features techniques
            tech_cols = [col for col in technical_features.columns 
                        if col not in price_cols + ['Volume', 'Returns']]
            
            if tech_cols:
                tech_features = technical_features[tech_cols]
                tech_features = tech_features.replace([np.inf, -np.inf], np.nan)
                tech_features = tech_features.ffill().bfill()
                normalized_features = self.feature_scaler.transform(tech_features)
            else:
                normalized_features = np.array([])
            
            # Création des séquences
            sequence_length = 15  # À ajuster selon vos besoins
            sequences = []
            
            for i in range(len(normalized_prices) - sequence_length + 1):
                sequence = np.concatenate([
                    normalized_prices[i:i+sequence_length],
                    normalized_features[i:i+sequence_length] if len(normalized_features) > 0 else np.array([])
                ], axis=1)
                sequences.append(sequence)
            
            return np.array(sequences)
            
        except Exception as e:
            self.logger.error(f"Erreur dans _prepare_live_features: {str(e)}")
            raise

    def _create_live_labels(self, new_data, horizon=12):
        """Crée les labels pour les nouvelles données"""
        try:
            future_returns = new_data['Close'].pct_change(periods=horizon).shift(-horizon)
            
            # Utiliser les mêmes seuils que pour l'entraînement
            pip_threshold = 0.0025
            returns = future_returns[:-horizon].values
            
            labels = np.zeros(len(returns))
            labels[returns > pip_threshold] = 2  # Hausse
            labels[returns < -pip_threshold] = 0  # Baisse
            labels[(returns >= -pip_threshold) & (returns <= pip_threshold)] = 1  # Neutre
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Erreur dans _create_live_labels: {str(e)}")
            raise

    def _validate_updated_sequences(self, X, y, min_class_ratio=0.1):
        """Valide la qualité des séquences mises à jour"""
        try:
            # Vérification des shapes
            if len(X) != len(y):
                self.logger.error("Incohérence dans les dimensions X/y")
                return False
            
            # Vérification des valeurs
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                self.logger.error("Valeurs invalides détectées dans X")
                return False
            
            # Vérification de la distribution des classes
            class_counts = np.bincount(y.astype(int))
            class_ratios = class_counts / len(y)
            
            if np.any(class_ratios < min_class_ratio):
                self.logger.warning("Distribution des classes déséquilibrée")
                self.logger.info("Ratios des classes:")
                for i, ratio in enumerate(class_ratios):
                    self.logger.info(f"• Classe {i}: {ratio:.2%}")
            
            # Vérification de la variance
            if np.var(X) < 1e-6:
                self.logger.error("Variance trop faible dans les features")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_updated_sequences: {str(e)}")
            return False