from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List, Generator
import logging
from datetime import datetime
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from collections import Counter
import warnings
import os
from src.features.technical_indicators import TechnicalIndicators
import gc

class BaseDataProcessor(ABC):
    """Classe de base abstraite pour le traitement des données"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.indicator_calculator = TechnicalIndicators()
        
        # Configuration des seuils par timeframe
        self.gap_thresholds = {
            '5m':  {'max_gap_ratio': 0.05, 'critical_ratio': 1.5},
            '15m': {'max_gap_ratio': 0.07, 'critical_ratio': 2.0},
            '30m': {'max_gap_ratio': 0.08, 'critical_ratio': 2.5},
            '1h':  {'max_gap_ratio': 0.10, 'critical_ratio': 3.0},
            '4h':  {'max_gap_ratio': 0.12, 'critical_ratio': 4.0},
            '1d':  {'max_gap_ratio': 0.15, 'critical_ratio': 5.0},
            '1w':  {'max_gap_ratio': 0.20, 'critical_ratio': 7.0},
            '1M':  {'max_gap_ratio': 0.25, 'critical_ratio': 10.0}
        }
        
        # Configuration du traitement par chunks
        self.chunk_config = {
            '5m':  {'size': 50000, 'overlap': 1000},  # ~6 mois de données 5m
            '15m': {'size': 25000, 'overlap': 500},
            '30m': {'size': 15000, 'overlap': 300},
            '1h':  {'size': 10000, 'overlap': 200},
            '4h':  {'size': 5000,  'overlap': 100},
            'default': {'size': 10000, 'overlap': 200}
        }
        
    def _setup_logging(self):
        """Configure le système de logging avec encodage UTF-8"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        
        # Handler pour fichier avec encodage UTF-8 explicite
        os.makedirs('logs/preprocessing', exist_ok=True)
        fh = logging.FileHandler(
            f'logs/preprocessing/processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'  # Spécifier l'encodage UTF-8
        )
        fh.setLevel(logging.DEBUG)
        
        # Handler pour console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter sans caractères Unicode spéciaux
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Remplacer les caractères Unicode par des alternatives ASCII
        def safe_format(record):
            record.msg = record.msg.replace('❌', 'X')  # Remplacer la croix rouge
            record.msg = record.msg.replace('⚠️', '!')  # Remplacer le warning
            record.msg = record.msg.replace('✓', '+')   # Remplacer le check
            return formatter.format(record)
        
        fh.formatter.format = safe_format
        ch.formatter.format = safe_format
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    @abstractmethod
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques avec gestion d'erreurs améliorée"""
        pass
    
    @abstractmethod
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte les patterns de prix spécifiques à l'instrument"""
        pass
        
    def detect_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Détecte et corrige les gaps avec validation"""
        try:
            self.logger.info(f"\nDétection vectorisée des gaps pour {timeframe}")
            df = df.copy()
            
            # Calcul de la volatilité pour ajuster les seuils
            volatility = self._calculate_volatility(df)
            thresholds = self._get_dynamic_thresholds(timeframe, volatility)
            
            # Conversion et calcul des intervalles
            df.index = pd.to_datetime(df.index)
            time_diffs = df.index.to_series().diff()
            expected_interval = pd.Timedelta(seconds=self._get_expected_interval(timeframe))
            
            # Détection vectorisée avec seuils dynamiques
            gap_mask = time_diffs > expected_interval * thresholds['min_gap_multiple']
            if not gap_mask.any():
                self.logger.info("✓ Aucun gap détecté")
                return df
            
            # Analyse des gaps avec seuils adaptés
            gap_sizes = time_diffs[gap_mask] / expected_interval
            gap_stats = {
                'small': ((gap_sizes > thresholds['min_gap_multiple']) & 
                         (gap_sizes <= thresholds['medium_gap_multiple'])).sum(),
                'medium': ((gap_sizes > thresholds['medium_gap_multiple']) & 
                          (gap_sizes <= thresholds['large_gap_multiple'])).sum(),
                'large': (gap_sizes > thresholds['large_gap_multiple']).sum()
            }
            
            self._log_gap_analysis(gap_stats, thresholds)
            
            # Traitement adaptatif selon la volatilité et le timeframe
            gap_percentage = gap_mask.sum() / len(df)
            max_allowed_ratio = thresholds['max_gap_ratio']
            
            if gap_percentage > max_allowed_ratio:
                self.logger.warning(
                    f"⚠️ Trop de gaps détectés ({gap_percentage:.1%} > {max_allowed_ratio:.1%}). "
                    "Traitement adaptatif appliqué"
                )
                return self._process_gaps_vectorized(
                    df, 
                    gap_mask & (gap_sizes > thresholds['critical_gap_multiple']),
                    timeframe,
                    thresholds=thresholds
                )
            
            # Traitement des gaps
            result = self._process_gaps_vectorized(df, gap_mask, timeframe, thresholds)
            
            # Validation post-traitement
            validation_ok, validation_stats = self._validate_data_quality(
                result, timeframe, thresholds
            )
            
            if not validation_ok:
                self.logger.warning("⚠️ Validation post-traitement échouée")
                self.logger.info("Tentative de retraitement avec paramètres plus stricts...")
                
                # Ajuster les seuils pour un second passage si nécessaire
                stricter_thresholds = {
                    k: v * 0.8 if isinstance(v, (int, float)) else v 
                    for k, v in thresholds.items()
                }
                
                result = self._process_gaps_vectorized(
                    result, gap_mask, timeframe, stricter_thresholds
                )
                
                # Validation finale
                final_ok, final_stats = self._validate_data_quality(
                    result, timeframe, stricter_thresholds
                )
                
                if not final_ok:
                    self.logger.error("❌ Échec de la correction des gaps")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur dans detect_gaps: {str(e)}")
            return df

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calcule la volatilité pour ajuster les seuils"""
        try:
            # Volatilité basée sur l'ATR normalisé
            atr = self.indicator_calculator.atr(
                df['High'], df['Low'], df['Close'], period=14
            )
            avg_price = df['Close'].mean()
            volatility_factor = (atr / avg_price).mean()
            
            # Normalisation entre 0.5 et 2
            return np.clip(volatility_factor * 100, 0.5, 2.0)
            
        except Exception as e:
            self.logger.warning(f"Erreur calcul volatilité: {str(e)}, utilisation valeur par défaut")
            return 1.0

    def _get_dynamic_thresholds(self, timeframe: str, volatility: float) -> Dict:
        """Calcule les seuils dynamiques selon le timeframe et la volatilité"""
        try:
            # Seuils de base pour ce timeframe
            base_thresholds = self.gap_thresholds.get(timeframe, self.gap_thresholds['1h'])
            
            # Ajustement selon la volatilité
            return {
                'min_gap_multiple': 1.5 * volatility,
                'medium_gap_multiple': 3.0 * volatility,
                'large_gap_multiple': 5.0 * volatility,
                'critical_gap_multiple': base_thresholds['critical_ratio'] * volatility,
                'max_gap_ratio': base_thresholds['max_gap_ratio'] * volatility,
                'interpolation_noise': 0.001 * volatility
            }
            
        except Exception as e:
            self.logger.error(f"Erreur seuils dynamiques: {str(e)}, utilisation valeurs par défaut")
            return {
                'min_gap_multiple': 1.5,
                'medium_gap_multiple': 3.0,
                'large_gap_multiple': 5.0,
                'critical_gap_multiple': 5.0,
                'max_gap_ratio': 0.1,
                'interpolation_noise': 0.001
            }

    def _log_gap_analysis(self, gap_stats: Dict, thresholds: Dict):
        """Log détaillé de l'analyse des gaps"""
        self.logger.info("\nAnalyse des gaps avec seuils dynamiques:")
        self.logger.info(f"• Petits gaps (x{thresholds['min_gap_multiple']:.1f}-"
                        f"x{thresholds['medium_gap_multiple']:.1f}): {gap_stats['small']}")
        self.logger.info(f"• Gaps moyens (x{thresholds['medium_gap_multiple']:.1f}-"
                        f"x{thresholds['large_gap_multiple']:.1f}): {gap_stats['medium']}")
        self.logger.info(f"• Grands gaps (>x{thresholds['large_gap_multiple']:.1f}): "
                        f"{gap_stats['large']}")
        self.logger.info(f"• Ratio maximum autorisé: {thresholds['max_gap_ratio']:.1%}")

    def _process_gaps_vectorized(self, df: pd.DataFrame, gap_mask: pd.Series, 
                               timeframe: str, thresholds: Dict) -> pd.DataFrame:
        """
        Traitement vectorisé des gaps
        """
        try:
            result = df.copy()
            gap_indices = gap_mask[gap_mask].index
            
            if len(gap_indices) == 0:
                return result
            
            # Préparation vectorisée des données pour interpolation
            gap_starts = df.index[df.index.get_indexer(gap_indices) - 1]
            gap_ends = gap_indices
            
            # Création vectorisée des nouveaux points temporels
            all_new_points = []
            all_interpolated_values = []
            
            # Traitement par lots pour optimiser la mémoire
            batch_size = 1000
            for batch_start in range(0, len(gap_starts), batch_size):
                batch_end = min(batch_start + batch_size, len(gap_starts))
                batch_gaps = list(zip(
                    gap_starts[batch_start:batch_end],
                    gap_ends[batch_start:batch_end]
                ))
                
                for start, end in batch_gaps:
                    # Création des points temporels
                    n_points = int((end - start).total_seconds() / 
                                 self._get_expected_interval(timeframe))
                    
                    new_points = pd.date_range(
                        start=start,
                        end=end,
                        periods=n_points + 2
                    )[1:-1]  # Exclure start et end qui existent déjà
                    
                    if len(new_points) == 0:
                        continue
                    
                    all_new_points.extend(new_points)
                    
                    # Interpolation vectorisée pour toutes les colonnes numériques
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    start_values = df.loc[start, numeric_cols]
                    end_values = df.loc[end, numeric_cols]
                    
                    # Calcul vectorisé des valeurs interpolées
                    steps = np.linspace(0, 1, len(new_points) + 2)[1:-1]
                    interpolated = np.outer(1 - steps, start_values) + np.outer(steps, end_values)
                    all_interpolated_values.extend(interpolated)
            
            if not all_new_points:
                return result
            
            # Création du DataFrame interpolé
            interpolated_df = pd.DataFrame(
                all_interpolated_values,
                index=all_new_points,
                columns=df.select_dtypes(include=[np.number]).columns
            )
            
            # Fusion vectorisée avec le DataFrame original
            result = pd.concat([result, interpolated_df]).sort_index()
            
            # Ajout de bruit gaussien vectorisé pour éviter les lignes identiques
            noise_scale = result.std() * thresholds['interpolation_noise']
            noise = np.random.normal(0, noise_scale, size=result.shape)
            result += noise
            
            # Vérifications finales vectorisées
            result = result.loc[~result.index.duplicated(keep='first')]
            time_diffs_after = result.index.to_series().diff()
            remaining_gaps = (time_diffs_after > expected_interval * thresholds['max_gap_ratio']).sum()
            
            self.logger.info(f"\nGaps restants après traitement: {remaining_gaps}")
            self.logger.info(f"Points interpolés ajoutés: {len(all_new_points)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur dans _process_gaps_vectorized: {str(e)}")
            return df

    def _get_expected_interval(self, timeframe: str) -> int:
        """
        Retourne l'intervalle attendu en secondes pour un timeframe
        """
        intervals = {
            '5m': 300,      # 5 minutes
            '15m': 900,     # 15 minutes
            '30m': 1800,    # 30 minutes
            '1h': 3600,     # 1 heure
            '4h': 14400,    # 4 heures
            '1d': 86400,    # 1 jour
            '1w': 604800,   # 1 semaine
            '1M': 2592000   # 30 jours (approximatif)
        }
        return intervals.get(timeframe, 300)  # Par défaut 5 minutes

    def detect_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte et corrige les anomalies de prix OHLC"""
        self.logger.info("Détection des anomalies de prix")
        df = df.copy()
        
        # Vérification des prix incohérents
        invalid_high = (df['High'] < df['Open']) | (df['High'] < df['Close'])
        invalid_low = (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
        
        anomalies = invalid_high | invalid_low
        if anomalies.sum() > 0:
            self.logger.warning(f"{anomalies.sum()} anomalies de prix détectées")
            
            # Correction par moyenne mobile
            df.loc[invalid_high, 'High'] = df[['Open', 'Close']].max(axis=1) * 1.001
            df.loc[invalid_low, 'Low'] = df[['Open', 'Close']].min(axis=1) * 0.999
            
            # Vérification après correction
            still_invalid = (df['High'] < df['Open']) | (df['High'] < df['Close']) | \
                           (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
            self.logger.info(f"Anomalies restantes après correction: {still_invalid.sum()}")
        
        return df

    def _handle_missing_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Gestion centralisée des données manquantes avec stratégies spécifiques par type
        
        Args:
            df: DataFrame à nettoyer
            timeframe: Timeframe des données pour ajuster les paramètres
            
        Returns:
            DataFrame nettoyé
        """
        self.logger.info(f"\nTraitement des données manquantes pour {timeframe}")
        df = df.copy()
        
        # 1. Analyse initiale des NaN
        nan_stats = df.isna().sum()
        if nan_stats.sum() == 0:
            self.logger.info("✓ Aucune donnée manquante détectée")
            return df
            
        self.logger.info("\nStatistiques des données manquantes:")
        for col in df.columns:
            nan_count = nan_stats[col]
            if nan_count > 0:
                nan_pct = (nan_count / len(df)) * 100
                self.logger.info(f"• {col}: {nan_count} NaN ({nan_pct:.2f}%)")
        
        # 2. Grouper les colonnes par type de données
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_cols = ['Volume']
        indicator_cols = [col for col in df.columns if col not in price_cols + volume_cols + ['Date']]
        
        # 3. Traitement des prix OHLC
        if any(df[price_cols].isna().any()):
            self.logger.info("\nTraitement des prix manquants...")
            
            # Identifier les gaps de prix
            price_gaps = df[price_cols].isna().all(axis=1)
            isolated_gaps = ~price_gaps.shift(1) & price_gaps & ~price_gaps.shift(-1)
            long_gaps = price_gaps & ~isolated_gaps
            
            # Traiter les gaps isolés par interpolation linéaire
            if isolated_gaps.any():
                self.logger.info(f"• Gaps isolés: {isolated_gaps.sum()}")
                for col in price_cols:
                    df.loc[isolated_gaps, col] = df[col].interpolate(
                        method='linear',
                        limit=1
                    )
            
            # Traiter les gaps longs avec plus de contexte
            if long_gaps.any():
                self.logger.info(f"• Gaps longs: {long_gaps.sum()}")
                window_size = self._get_window_size(timeframe)
                
                for col in price_cols:
                    # Interpolation avec fenêtre adaptative
                    df[col] = self._adaptive_interpolation(
                        df[col],
                        window_size=window_size,
                        timeframe=timeframe
                    )
        
        # 4. Traitement du volume
        if df[volume_cols].isna().any().any():
            self.logger.info("\nTraitement des volumes manquants...")
            
            # Calculer la moyenne mobile du volume
            vol_ma = df['Volume'].rolling(
                window=self._get_window_size(timeframe),
                min_periods=1
            ).mean()
            
            # Remplacer les NaN par la moyenne locale
            vol_mask = df['Volume'].isna()
            df.loc[vol_mask, 'Volume'] = vol_ma[vol_mask]
            
            # Ajouter un bruit minimal pour éviter les valeurs identiques
            noise = np.random.normal(
                0,
                vol_ma.std() * 0.01,
                size=len(df[vol_mask])
            )
            df.loc[vol_mask, 'Volume'] += noise
            
            # S'assurer que le volume reste positif
            df['Volume'] = df['Volume'].clip(lower=0)
        
        # 5. Traitement des indicateurs techniques
        if indicator_cols and df[indicator_cols].isna().any().any():
            self.logger.info("\nTraitement des indicateurs manquants...")
            
            for col in indicator_cols:
                nan_mask = df[col].isna()
                if not nan_mask.any():
                    continue
                
                # Déterminer la stratégie de remplissage
                if 'SMA' in col or 'EMA' in col:
                    # Moyennes mobiles : interpolation
                    df[col] = df[col].interpolate(method='linear', limit=window_size)
                elif 'RSI' in col or 'MACD' in col:
                    # Oscillateurs : forward fill puis backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # Autres indicateurs : interpolation avec limite
                    df[col] = self._adaptive_interpolation(
                        df[col],
                        window_size=window_size,
                        timeframe=timeframe
                    )
        
        # 6. Vérification finale
        remaining_nans = df.isna().sum()
        if remaining_nans.sum() > 0:
            self.logger.warning("\n⚠️ NaN restants après traitement:")
            for col in df.columns:
                if remaining_nans[col] > 0:
                    self.logger.warning(f"• {col}: {remaining_nans[col]} NaN")
            
            # Dernière tentative de nettoyage
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            if df.isna().sum().sum() > 0:
                raise ValueError("Impossible de nettoyer toutes les données manquantes")
        else:
            self.logger.info("\n✓ Toutes les données manquantes ont été traitées")
        
        return df
    
    def _get_window_size(self, timeframe: str) -> int:
        """Retourne la taille de fenêtre adaptée au timeframe"""
        window_sizes = {
            '5m': 12,    # 1 heure
            '15m': 16,   # 4 heures
            '30m': 16,   # 8 heures
            '1h': 24,    # 1 jour
            '4h': 30,    # 5 jours
            '1d': 20,    # 1 mois
            '1w': 12,    # 3 mois
            '1M': 6      # 6 mois
        }
        return window_sizes.get(timeframe, 20)
    
    def _adaptive_interpolation(self, series: pd.Series, window_size: int, timeframe: str) -> pd.Series:
        """
        Interpolation adaptative vectorisée avec vérification de la cohérence
        
        Args:
            series: Série à interpoler
            window_size: Taille de la fenêtre
            timeframe: Timeframe des données
            
        Returns:
            Série interpolée
        """
        result = series.copy()
        nan_mask = series.isna()
        
        if not nan_mask.any():
            return result
        
        # Identifier les segments NaN en une seule passe
        changes = nan_mask.astype(int).diff().fillna(0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(ends) < len(starts):  # Si le dernier segment continue jusqu'à la fin
            ends = np.append(ends, len(series))
        
        # Calculer les statistiques une seule fois
        series_std = series.std()
        series_mean = series.mean()
        
        # Traiter tous les segments d'un coup
        segments = np.column_stack((starts, ends))
        segment_lengths = ends - starts
        
        # Séparer les segments courts et longs
        short_mask = segment_lengths <= window_size
        long_mask = ~short_mask
        
        # Traiter les segments courts
        if short_mask.any():
            short_segments = segments[short_mask]
            for start, end in short_segments:
                # Utiliser les valeurs valides avant et après
                left_val = series.iloc[start-1] if start > 0 else series_mean
                right_val = series.iloc[end] if end < len(series) else series_mean
                
                # Interpolation linéaire vectorisée
                segment_length = end - start
                result.iloc[start:end] = np.linspace(left_val, right_val, segment_length)
        
        # Traiter les segments longs
        if long_mask.any():
            long_segments = segments[long_mask]
            for start, end in long_segments:
                segment_length = end - start
                
                # Calculer la tendance locale
                window_before = slice(max(0, start-window_size), start)
                window_after = slice(end, min(end+window_size, len(series)))
                
                before_stats = series.iloc[window_before].agg(['mean', 'std'])
                after_stats = series.iloc[window_after].agg(['mean', 'std'])
                
                # Calculer les paramètres d'interpolation
                local_mean = np.mean([before_stats['mean'], after_stats['mean']])
                local_std = np.mean([before_stats['std'], after_stats['std']])
                trend = (after_stats['mean'] - before_stats['mean']) / segment_length
                
                # Générer l'interpolation vectorisée
                x = np.arange(segment_length)
                base = local_mean + trend * x
                
                # Ajouter de la volatilité décroissante
                volatility = local_std * np.exp(-x/window_size)
                noise = np.random.normal(0, volatility)
                
                # Combiner les composantes
                result.iloc[start:end] = base + noise
        
        # Vérification finale des bornes
        if series.min() is not None and series.max() is not None:
            result = result.clip(series.min(), series.max())
        
        return result

    def _interpolate_long_segment(self, series: pd.Series, start: int, end: int,
                                window_size: int, timeframe: str) -> pd.Series:
        """
        Interpolation vectorisée pour les longs segments
        """
        segment_length = end - start
        
        # Extraire les statistiques locales en une fois
        before_window = slice(max(0, start-window_size), start)
        after_window = slice(end, min(end+window_size, len(series)))
        
        before_stats = series.iloc[before_window].agg(['mean', 'std'])
        after_stats = series.iloc[after_window].agg(['mean', 'std'])
        
        # Calculer les paramètres d'interpolation
        local_mean = np.mean([before_stats['mean'], after_stats['mean']])
        local_std = np.mean([before_stats['std'], after_stats['std']])
        trend = (after_stats['mean'] - before_stats['mean']) / segment_length
        
        # Générer l'interpolation vectorisée
        x = np.arange(segment_length)
        base = local_mean + trend * x
        
        # Ajouter de la volatilité décroissante
        volatility = local_std * np.exp(-x/window_size)
        noise = np.random.normal(0, volatility)
        
        # Combiner les composantes
        result = base + noise
        
        # Assurer la continuité aux extrémités
        if start > 0:
            result[0] = series.iloc[start-1]
        if end < len(series):
            result[-1] = series.iloc[end]
        
        return result

    def _identify_nan_segments(self, series: pd.Series) -> np.ndarray:
        """
        Identifie les segments NaN de manière vectorisée
        
        Returns:
            array: Segments sous forme [[start1, end1], [start2, end2], ...]
        """
        # Convertir le masque NaN en entiers
        nan_ints = series.isna().astype(int)
        
        # Trouver les changements
        changes = np.diff(np.concatenate(([0], nan_ints, [0])))
        
        # Extraire les indices de début et fin
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        return np.column_stack((starts, ends))

    # Constantes pour la gestion des gaps
    GAP_TOLERANCES = {
        '5m':  {'multiplier': 20, 'max_pct': 0.002},  # 0.2% max
        '15m': {'multiplier': 25, 'max_pct': 0.003},  # 0.3% max
        '30m': {'multiplier': 30, 'max_pct': 0.004},  # 0.4% max
        '1h':  {'multiplier': 40, 'max_pct': 0.005},  # 0.5% max
        '4h':  {'multiplier': 50, 'max_pct': 0.007},  # 0.7% max
        '1d':  {'multiplier': 60, 'max_pct': 0.01},   # 1.0% max
        '1w':  {'multiplier': 80, 'max_pct': 0.015},  # 1.5% max
        '1M':  {'multiplier': 100, 'max_pct': 0.02}   # 2.0% max
    }
    
    def _get_timeframe_tolerances(self, timeframe: str) -> Tuple[float, float, float]:
        """
        Calcule les tolérances adaptatives pour un timeframe donné
        
        Args:
            timeframe: Timeframe des données
            
        Returns:
            Tuple[float, float, float]: (multiplicateur ATR, ATR minimum, tolérance de base)
        """
        # Obtenir la configuration du timeframe
        config = self.GAP_TOLERANCES.get(timeframe, {'multiplier': 30, 'max_pct': 0.005})
        
        return (
            config['multiplier'],
            config['max_pct'],
            config['multiplier'] * 0.1  # Tolérance de base = 10% du multiplicateur
        )
    
    def _verify_temporal_continuity(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """
        Vérifie et corrige la continuité temporelle des données avec des tolérances strictes
        
        Args:
            df: DataFrame à vérifier
            timeframe: Timeframe des données
            
        Returns:
            Tuple[DataFrame, bool]: (DataFrame corrigé, indicateur de modifications)
        """
        self.logger.info(f"\nVérification de la continuité temporelle pour {timeframe}")
        df = df.copy()
        modified = False
        
        # Calculer les métriques de référence
        median_price = df['Close'].median()
        atr = self._calculate_atr(df)
        
        # Obtenir les tolérances pour ce timeframe
        gap_multiplier, max_pct, base_tolerance = self._get_timeframe_tolerances(timeframe)
        
        # Calculer la tolérance adaptative avec limite absolue
        adaptive_tolerance = np.minimum(
            atr * gap_multiplier,  # Limite basée sur l'ATR
            median_price * max_pct  # Limite absolue en pourcentage
        )
        
        # Détecter les discontinuités
        price_changes = df['Close'].diff().abs()
        volume_changes = df['Volume'].pct_change().abs()
        
        # Identifier les anomalies
        price_anomalies = price_changes > adaptive_tolerance
        volume_spikes = volume_changes > 5.0  # 500% de changement
        
        if price_anomalies.any() or volume_spikes.any():
            self.logger.warning(f"\nAnomalies détectées dans {timeframe}:")
            self.logger.warning(f"• Anomalies de prix: {price_anomalies.sum()}")
            self.logger.warning(f"• Pics de volume: {volume_spikes.sum()}")
            
            # Statistiques détaillées
            if price_anomalies.any():
                anomaly_stats = price_changes[price_anomalies].describe()
                self.logger.info("\nStatistiques des variations anormales:")
                self.logger.info(f"\n{anomaly_stats}")
            
            # Vérifier si les anomalies sont trop nombreuses
            anomaly_threshold = len(df) * 0.01  # Max 1% de données anormales
            if price_anomalies.sum() > anomaly_threshold:
                self.logger.error(
                    f"Trop d'anomalies détectées: {price_anomalies.sum()} "
                    f"(seuil: {anomaly_threshold:.0f})"
                )
                raise ValueError(f"Données {timeframe} trop incohérentes")
            
            # Corriger les anomalies isolées
            isolated_anomalies = price_anomalies & ~price_anomalies.shift(1) & ~price_anomalies.shift(-1)
            if isolated_anomalies.any():
                self.logger.info(f"Correction de {isolated_anomalies.sum()} anomalies isolées")
                
                # Interpolation locale pour les anomalies isolées
                for col in ['Open', 'High', 'Low', 'Close']:
                    df.loc[isolated_anomalies, col] = df[col].interpolate(
                        method='linear',
                        limit=1
                    )
                
                modified = True
            
            # Marquer les segments suspects pour analyse
            suspicious_segments = self._identify_suspicious_segments(
                df, price_anomalies, adaptive_tolerance
            )
            
            if suspicious_segments:
                self.logger.warning("\nSegments suspects détectés:")
                for start, end in suspicious_segments:
                    self.logger.warning(
                        f"• {df.index[start]} → {df.index[end]}: "
                        f"{end - start + 1} points"
                    )
        
        return df, modified
    
    def _identify_suspicious_segments(self, df: pd.DataFrame, 
                                   anomalies: pd.Series,
                                   tolerance: float) -> List[Tuple[int, int]]:
        """
        Identifie les segments de données suspects pour analyse manuelle
        
        Args:
            df: DataFrame des données
            anomalies: Série booléenne des anomalies
            tolerance: Tolérance maximale
            
        Returns:
            Liste de tuples (début, fin) des segments suspects
        """
        segments = []
        start = None
        
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and start is None:
                start = i
            elif not is_anomaly and start is not None:
                if i - start > 1:  # Ignorer les anomalies isolées
                    segments.append((start, i-1))
                start = None
        
        if start is not None and start < len(anomalies) - 1:
            segments.append((start, len(anomalies) - 1))
        
        return segments

    def _balance_labels(self, X: np.ndarray, y: np.ndarray, method='smart') -> Tuple[np.ndarray, np.ndarray]:
        """
        Équilibre les classes avec validation des échantillons synthétiques
        
        Args:
            X: Features (shape: [n_samples, sequence_length, n_features])
            y: Labels
            method: Méthode d'équilibrage ('smart' ou 'simple')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X équilibré, y équilibré)
        """
        self.logger.info("\n🔄 Équilibrage des classes...")
        
        # Statistiques initiales
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        self.logger.info("\nDistribution initiale:")
        for label, count in sorted(class_dist.items()):
            pct = count / len(y) * 100
            self.logger.info(f"• Classe {label}: {count} ({pct:.1f}%)")
        
        if method == 'smart':
            try:
                # Reshape pour SMOTE
                n_samples, seq_length, n_features = X.shape
                X_reshaped = X.reshape(n_samples, -1)
                
                # Configuration SMOTE adaptative
                sampling_strategy = self._get_sampling_strategy(class_dist)
                k_neighbors = min(5, min(counts) - 1)  # Éviter l'erreur de voisinage
                
                # Équilibrage avec SMOTEENN
                smote = SMOTEENN(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    n_neighbors=k_neighbors
                )
                
                X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
                
                # Reshape retour à la forme originale
                X_balanced = X_resampled.reshape(-1, seq_length, n_features)
                
                # Validation des échantillons synthétiques
                X_balanced = self._validate_synthetic_samples(X_balanced, X)
                
            except Exception as e:
                self.logger.warning(f"\n⚠️ Erreur SMOTE: {str(e)}")
                self.logger.info("→ Utilisation de la méthode simple")
                return self._balance_simple(X, y)
        else:
            X_balanced, y_resampled = self._balance_simple(X, y)
        
        # Statistiques finales
        final_dist = dict(zip(*np.unique(y_resampled, return_counts=True)))
        
        self.logger.info("\nDistribution finale:")
        for label, count in sorted(final_dist.items()):
            pct = count / len(y_resampled) * 100
            self.logger.info(f"• Classe {label}: {count} ({pct:.1f}%)")
        
        return X_balanced, y_resampled

    def _validate_synthetic_samples(self, synthetic: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Valide et corrige les échantillons synthétiques
        
        Args:
            synthetic: Données synthétiques générées
            original: Données originales de référence
            
        Returns:
            np.ndarray: Données synthétiques validées
        """
        self.logger.info("\nValidation des échantillons synthétiques...")
        
        # Indices des différentes colonnes
        price_cols = slice(0, 4)  # OHLC
        volume_col = 4
        indicator_cols = slice(5, None)  # Autres indicateurs
        
        # 1. Validation des prix OHLC
        min_price = np.min(original[:, :, price_cols]) * 0.9
        max_price = np.max(original[:, :, price_cols]) * 1.1
        
        # Détecter les anomalies de prix
        price_anomalies = (synthetic[:, :, price_cols] < min_price) | (synthetic[:, :, price_cols] > max_price)
        n_price_anomalies = np.sum(price_anomalies)
        
        if n_price_anomalies > 0:
            self.logger.warning(f"⚠️ {n_price_anomalies} anomalies de prix détectées")
            synthetic[:, :, price_cols] = np.clip(synthetic[:, :, price_cols], min_price, max_price)
        
        # 2. Validation de la cohérence OHLC
        for i in range(len(synthetic)):
            for t in range(synthetic.shape[1]):
                ohlc = synthetic[i, t, :4]
                # Assurer High >= max(Open, Close)
                synthetic[i, t, 2] = max(ohlc[0], ohlc[3], ohlc[2])  # High
                # Assurer Low <= min(Open, Close)
                synthetic[i, t, 1] = min(ohlc[0], ohlc[3], ohlc[1])  # Low
        
        # 3. Validation du volume
        min_volume = np.min(original[:, :, volume_col]) * 0.5
        max_volume = np.max(original[:, :, volume_col]) * 2.0
        
        volume_anomalies = (synthetic[:, :, volume_col] < min_volume) | (synthetic[:, :, volume_col] > max_volume)
        if np.any(volume_anomalies):
            self.logger.warning(f"⚠️ {np.sum(volume_anomalies)} anomalies de volume détectées")
            synthetic[:, :, volume_col] = np.clip(synthetic[:, :, volume_col], min_volume, max_volume)
        
        # 4. Validation des indicateurs techniques
        if synthetic.shape[2] > 5:  # S'il y a des indicateurs
            for i in range(5, synthetic.shape[2]):
                indicator = synthetic[:, :, i]
                orig_indicator = original[:, :, i]
                
                # Calculer les limites basées sur les données originales
                indicator_min = np.min(orig_indicator) * 0.9
                indicator_max = np.max(orig_indicator) * 1.1
                
                # Corriger les valeurs aberrantes
                synthetic[:, :, i] = np.clip(indicator, indicator_min, indicator_max)
        
        # 5. Vérification de la continuité temporelle
        for i in range(len(synthetic)):
            # Éviter les sauts de prix trop importants
            price_changes = np.diff(synthetic[i, :, 3])  # Variations des prix de clôture
            max_change = np.std(original[:, :, 3]) * 3  # 3 écarts-types
            
            large_jumps = np.abs(price_changes) > max_change
            if np.any(large_jumps):
                self.logger.warning(f"⚠️ Variations importantes détectées dans la séquence {i}")
                # Lisser les variations extrêmes
                synthetic[i, 1:, 3][large_jumps] = synthetic[i, :-1, 3][large_jumps] + \
                                                 np.sign(price_changes[large_jumps]) * max_change
        
        self.logger.info("✓ Validation terminée")
        return synthetic

    def _get_sampling_strategy(self, class_dist: Dict[int, int]) -> Dict[int, int]:
        """
        Détermine la stratégie d'échantillonnage optimale
        """
        max_samples = max(class_dist.values())
        strategy = {}
        
        for label, count in class_dist.items():
            if count < max_samples:
                # Limiter l'augmentation à 3x pour éviter le surapprentissage
                strategy[label] = min(max_samples, count * 3)
        
        return strategy

    def _validate_temporal_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Valide et nettoie les données temporelles
        
        Args:
            df: DataFrame à valider
            timeframe: Timeframe des données
            
        Returns:
            DataFrame validé et nettoyé
        """
        try:
            df = df.copy()
            
            # 1. Vérification de l'index temporel
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("Index non temporel détecté, conversion...")
                df.index = pd.to_datetime(df.index)
            
            # 2. Détection et suppression des doublons
            duplicates = df.index.duplicated()
            if duplicates.any():
                n_duplicates = duplicates.sum()
                self.logger.warning(
                    f"Doublons détectés dans l'index ({n_duplicates}), suppression..."
                )
                df = df[~duplicates]
                self.logger.info(f"→ {n_duplicates} doublons supprimés")
            
            # 3. Vérification de l'ordre temporel
            if not df.index.is_monotonic_increasing:
                self.logger.warning("Index non trié, tri chronologique...")
                df = df.sort_index()
            
            # 4. Vérification des intervalles
            time_diffs = df.index.to_series().diff()
            expected_interval = pd.Timedelta(self._get_expected_interval(timeframe), unit='s')
            irregular_intervals = time_diffs != expected_interval
            
            if irregular_intervals.any():
                n_irregular = irregular_intervals.sum()
                self.logger.warning(
                    f"Intervalles irréguliers détectés ({n_irregular})"
                )
                
                # Log des statistiques des intervalles
                interval_stats = time_diffs.describe()
                self.logger.info("\nStatistiques des intervalles:")
                self.logger.info(f"• Minimum: {interval_stats['min']}")
                self.logger.info(f"• Maximum: {interval_stats['max']}")
                self.logger.info(f"• Moyenne: {interval_stats['mean']}")
                
                # Correction si nécessaire
                if n_irregular > len(df) * 0.01:  # Plus de 1% d'irrégularités
                    self.logger.warning("Rééchantillonnage des données...")
                    df = self._resample_data(df, timeframe)
            
            # 5. Vérification finale
            self.logger.info("\nStatistiques après validation:")
            self.logger.info(f"• Lignes: {len(df)}")
            self.logger.info(f"• Période: {df.index.min()} → {df.index.max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_temporal_data: {str(e)}")
            raise

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Rééchantillonne les données selon le timeframe
        
        Args:
            df: DataFrame à rééchantillonner
            timeframe: Timeframe cible
            
        Returns:
            DataFrame rééchantillonné
        """
        try:
            # Convertir le timeframe en format pandas
            freq = self._timeframe_to_freq(timeframe)
            
            # Rééchantillonnage avec règles spécifiques pour OHLCV
            resampled = pd.DataFrame()
            resampled['Open'] = df['Open'].resample(freq).first()
            resampled['High'] = df['High'].resample(freq).max()
            resampled['Low'] = df['Low'].resample(freq).min()
            resampled['Close'] = df['Close'].resample(freq).last()
            resampled['Volume'] = df['Volume'].resample(freq).sum()
            
            # Traitement des autres colonnes
            for col in df.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    resampled[col] = df[col].resample(freq).mean()
            
            # Supprimer les lignes avec NaN
            resampled = resampled.dropna()
            
            self.logger.info(
                f"Rééchantillonnage terminé: {len(df)} → {len(resampled)} lignes"
            )
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Erreur dans _resample_data: {str(e)}")
            raise

    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convertit le timeframe en fréquence pandas"""
        mapping = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }
        return mapping.get(timeframe, '1D')  # Par défaut 1 jour

    def prepare_training_data(self, data_dict: Dict[str, pd.DataFrame], 
                            sequence_length: int = 15,
                            prediction_horizon: int = 12,
                            target_samples: int = 3000,
                            pip_threshold: float = 0.0025) -> Dict[str, np.ndarray]:
        """Prépare les données pour l'entraînement avec validation temporelle"""
        
        # Validation temporelle pour chaque timeframe
        validated_data = {}
        for tf, df in data_dict.items():
            try:
                validated_df = self._validate_temporal_data(df, tf)
                validated_data[tf] = validated_df
            except Exception as e:
                self.logger.error(f"Erreur lors de la validation de {tf}: {str(e)}")
                continue
        
        if not validated_data:
            raise ValueError("Aucune donnée valide après validation temporelle")
        
        # Au lieu d'appeler super(), implémenter directement le traitement ici
        processed_data = {}
        for tf, df in validated_data.items():
            try:
                # Ajouter les indicateurs techniques
                df_processed = self.add_technical_indicators(df)
                
                # Ajouter les patterns de prix
                df_processed = self.add_price_patterns(df_processed)
                
                # Convertir en numpy array et normaliser si nécessaire
                processed_data[tf] = self._prepare_sequences(
                    df_processed,
                    sequence_length,
                    prediction_horizon,
                    target_samples,
                    pip_threshold
                )
                
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {tf}: {str(e)}")
                continue
            
        return processed_data

    def _prepare_sequences(self, df: pd.DataFrame,
                         sequence_length: int,
                         prediction_horizon: int,
                         target_samples: int,
                         pip_threshold: float) -> Dict[str, np.ndarray]:
        """Prépare les séquences pour l'entraînement"""
        # Implémentation de la préparation des séquences
        # Cette méthode doit être implémentée selon vos besoins spécifiques
        pass

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute uniquement les indicateurs de base en cas d'erreur"""
        try:
            self.logger.warning("\n⚠️ Utilisation des indicateurs de base uniquement")
            df = df.copy()
            
            # Utiliser TechnicalIndicators avec configuration minimale
            basic_params = {
                'sma': [20],
                'ema': [],
                'rsi': [14],
                'macd': None,
                'bbands': None,
                'stoch': None,
                'atr': [14],
                'adx': None,
                'momentum': None,
                'volume_ma': [20]
            }
            
            return self.indicator_calculator.calculate_all(df, params=basic_params)
            
        except Exception as e:
            self.logger.error(f"Erreur dans _add_basic_indicators: {str(e)}")
            return df

    def _validate_data_quality(self, df: pd.DataFrame, timeframe: str, 
                         thresholds: Dict) -> Tuple[bool, Dict]:
        """
        Valide la qualité des données après traitement
        
        Returns:
            Tuple[bool, Dict]: (validation_ok, statistics)
        """
        try:
            stats = {
                'gaps': self._validate_gaps(df, timeframe, thresholds),
                'prices': self._validate_prices(df),
                'continuity': self._validate_continuity(df, timeframe)
            }
            
            # Vérification globale
            validation_ok = all(
                stat.get('valid', False) 
                for stat in stats.values()
            )
            
            # Log des résultats
            self._log_validation_results(stats, timeframe)
            
            return validation_ok, stats
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_data_quality: {str(e)}")
            return False, {}

    def _validate_gaps(self, df: pd.DataFrame, timeframe: str, 
                      thresholds: Dict) -> Dict:
        """Valide la correction des gaps"""
        try:
            time_diffs = df.index.to_series().diff()
            expected_interval = pd.Timedelta(seconds=self._get_expected_interval(timeframe))
            
            # Analyse des gaps résiduels
            gap_sizes = time_diffs / expected_interval
            gap_stats = {
                'small': ((gap_sizes > thresholds['min_gap_multiple']) & 
                         (gap_sizes <= thresholds['medium_gap_multiple'])).sum(),
                'medium': ((gap_sizes > thresholds['medium_gap_multiple']) & 
                          (gap_sizes <= thresholds['large_gap_multiple'])).sum(),
                'large': (gap_sizes > thresholds['large_gap_multiple']).sum()
            }
            
            total_gaps = sum(gap_stats.values())
            gap_ratio = total_gaps / len(df)
            
            return {
                'valid': gap_ratio <= thresholds['max_gap_ratio'],
                'gap_ratio': gap_ratio,
                'gap_stats': gap_stats,
                'max_gap_size': gap_sizes.max(),
                'avg_gap_size': gap_sizes[gap_sizes > 1].mean()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_gaps: {str(e)}")
            return {'valid': False, 'error': str(e)}

    def _validate_prices(self, df: pd.DataFrame) -> Dict:
        """Valide la cohérence des prix OHLC"""
        try:
            # Vérifications de base
            invalid_high = (df['High'] < df['Open']) | (df['High'] < df['Close'])
            invalid_low = (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
            zero_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
            
            # Vérifications avancées
            price_jumps = df['Close'].pct_change().abs()
            extreme_moves = price_jumps > 0.1  # Mouvements > 10%
            
            # Calcul des statistiques
            stats = {
                'valid': not (invalid_high.any() or invalid_low.any() or zero_prices.any()),
                'invalid_high_count': invalid_high.sum(),
                'invalid_low_count': invalid_low.sum(),
                'zero_prices_count': zero_prices.sum(),
                'extreme_moves_count': extreme_moves.sum(),
                'max_price_jump': price_jumps.max(),
                'price_volatility': price_jumps.std()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_prices: {str(e)}")
            return {'valid': False, 'error': str(e)}

    def _validate_continuity(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Valide la continuité et la qualité des données"""
        try:
            # Analyse de la distribution temporelle
            time_diffs = df.index.to_series().diff()
            expected_interval = pd.Timedelta(seconds=self._get_expected_interval(timeframe))
            
            # Calcul des statistiques de continuité
            stats = {
                'total_points': len(df),
                'unique_points': len(df.index.unique()),
                'duplicates': len(df) - len(df.index.unique()),
                'missing_values': df.isna().sum().to_dict(),
                'time_coverage': {
                    'start': df.index.min(),
                    'end': df.index.max(),
                    'expected_points': int((df.index.max() - df.index.min()) / expected_interval),
                    'actual_points': len(df)
                }
            }
            
            # Validation de la couverture
            coverage_ratio = len(df) / stats['time_coverage']['expected_points']
            stats['coverage_ratio'] = coverage_ratio
            stats['valid'] = (coverage_ratio >= 0.95)  # Au moins 95% de couverture
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_continuity: {str(e)}")
            return {'valid': False, 'error': str(e)}

    def _log_validation_results(self, stats: Dict, timeframe: str):
        """Log détaillé des résultats de validation"""
        self.logger.info(f"\nRésultats de validation pour {timeframe}:")
        
        # 1. Gaps
        if 'gaps' in stats:
            gap_stats = stats['gaps']
            self.logger.info("\nValidation des gaps:")
            self.logger.info(f"• Ratio de gaps: {gap_stats['gap_ratio']:.2%}")
            self.logger.info("• Distribution des gaps:")
            for gap_type, count in gap_stats['gap_stats'].items():
                self.logger.info(f"  - {gap_type}: {count}")
            self.logger.info(f"• Plus grand gap: x{gap_stats['max_gap_size']:.1f}")
            status = "✓" if gap_stats['valid'] else "⚠️"
            self.logger.info(f"{status} Validation gaps: {'OK' if gap_stats['valid'] else 'NOK'}")
        
        # 2. Prix
        if 'prices' in stats:
            price_stats = stats['prices']
            self.logger.info("\nValidation des prix:")
            self.logger.info(f"• Prix High invalides: {price_stats['invalid_high_count']}")
            self.logger.info(f"• Prix Low invalides: {price_stats['invalid_low_count']}")
            self.logger.info(f"• Mouvements extrêmes: {price_stats['extreme_moves_count']}")
            self.logger.info(f"• Volatilité: {price_stats['price_volatility']:.4f}")
            status = "✓" if price_stats['valid'] else "⚠️"
            self.logger.info(f"{status} Validation prix: {'OK' if price_stats['valid'] else 'NOK'}")
        
        # 3. Continuité
        if 'continuity' in stats:
            cont_stats = stats['continuity']
            self.logger.info("\nValidation de la continuité:")
            self.logger.info(f"• Points totaux: {cont_stats['total_points']}")
            self.logger.info(f"• Doublons: {cont_stats['duplicates']}")
            self.logger.info(f"• Couverture: {cont_stats['coverage_ratio']:.2%}")
            status = "✓" if cont_stats['valid'] else "⚠️"
            self.logger.info(f"{status} Validation continuité: {'OK' if cont_stats['valid'] else 'NOK'}")

    def process_large_dataset(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Traite un grand dataset par chunks avec gestion optimisée de la mémoire
        """
        try:
            self.logger.info(f"\nTraitement par chunks pour {timeframe}")
            
            # Configuration du chunk
            chunk_params = self.chunk_config.get(timeframe, self.chunk_config['default'])
            chunk_size = chunk_params['size']
            overlap = chunk_params['overlap']
            
            # Estimation de la mémoire
            estimated_memory = self._estimate_memory_usage(df, chunk_size)
            self.logger.info(f"Mémoire estimée par chunk: {estimated_memory:.2f} MB")
            
            # Traitement par chunks
            processed_chunks = []
            total_chunks = (len(df) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(df), chunk_size - overlap):
                chunk_num = (i // (chunk_size - overlap)) + 1
                self.logger.info(f"\nTraitement chunk {chunk_num}/{total_chunks}")
                
                # Extraction du chunk avec chevauchement
                chunk_end = min(i + chunk_size, len(df))
                chunk = df.iloc[i:chunk_end].copy()
                
                # Traitement du chunk
                processed_chunk = self._process_chunk(chunk, timeframe)
                
                # Gestion du chevauchement
                if processed_chunks and i > 0:
                    # Fusionner la zone de chevauchement
                    processed_chunk = self._merge_overlap(
                        processed_chunks[-1],
                        processed_chunk,
                        overlap
                    )
                    # Retirer la partie chevauchante du chunk précédent
                    processed_chunks[-1] = processed_chunks[-1].iloc[:-overlap]
                
                processed_chunks.append(processed_chunk)
                
                # Nettoyage explicite de la mémoire
                del chunk
                gc.collect()
            
            # Concaténation finale
            result = pd.concat(processed_chunks)
            
            # Validation finale
            self._validate_merged_result(result, df, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur dans process_large_dataset: {str(e)}")
            return df

    def _process_chunk(self, chunk: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Traite un chunk individuel"""
        try:
            # 1. Détection et correction des gaps
            chunk = self.detect_gaps(chunk, timeframe)
            
            # 2. Calcul des indicateurs techniques
            chunk = self.add_technical_indicators(chunk)
            
            # 3. Détection des patterns de prix
            chunk = self.add_price_patterns(chunk)
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"Erreur dans _process_chunk: {str(e)}")
            return chunk

    def _merge_overlap(self, prev_chunk: pd.DataFrame, 
                      curr_chunk: pd.DataFrame,
                      overlap: int) -> pd.DataFrame:
        """Fusionne la zone de chevauchement entre deux chunks"""
        try:
            # Identifier la zone de chevauchement
            overlap_start = curr_chunk.index[0]
            overlap_end = curr_chunk.index[overlap-1]
            
            # Calculer les moyennes pondérées pour la transition
            weights = np.linspace(0, 1, overlap)
            overlap_mask = (curr_chunk.index >= overlap_start) & (curr_chunk.index <= overlap_end)
            
            # Fusion progressive des valeurs
            numeric_cols = curr_chunk.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                curr_vals = curr_chunk.loc[overlap_mask, col].values
                prev_vals = prev_chunk.loc[overlap_mask, col].values
                
                # Interpolation pondérée
                curr_chunk.loc[overlap_mask, col] = (
                    prev_vals * (1 - weights) + curr_vals * weights
                )
            
            return curr_chunk
            
        except Exception as e:
            self.logger.error(f"Erreur dans _merge_overlap: {str(e)}")
            return curr_chunk

    def _estimate_memory_usage(self, df: pd.DataFrame, chunk_size: int) -> float:
        """Estime l'utilisation mémoire d'un chunk en MB"""
        try:
            # Calculer la taille d'un échantillon
            sample_size = min(1000, len(df))
            sample_memory = df.head(sample_size).memory_usage(deep=True).sum() / sample_size
            
            # Estimer la taille du chunk
            estimated_chunk_memory = (sample_memory * chunk_size) / (1024 * 1024)  # En MB
            
            return estimated_chunk_memory
            
        except Exception as e:
            self.logger.error(f"Erreur dans _estimate_memory_usage: {str(e)}")
            return 0.0

    def _validate_merged_result(self, result: pd.DataFrame, 
                              original: pd.DataFrame,
                              timeframe: str):
        """Valide le résultat final après fusion des chunks"""
        try:
            # Vérifications de base
            self.logger.info("\nValidation du résultat final:")
            self.logger.info(f"• Points originaux: {len(original)}")
            self.logger.info(f"• Points traités: {len(result)}")
            
            # Vérifier la continuité temporelle
            time_diffs = result.index.to_series().diff()
            expected_interval = pd.Timedelta(seconds=self._get_expected_interval(timeframe))
            gaps = (time_diffs > expected_interval * 1.5).sum()
            
            self.logger.info(f"• Gaps résiduels: {gaps}")
            
            # Vérifier la cohérence des données
            validation_ok, stats = self._validate_data_quality(result, timeframe, 
                                                             self._get_dynamic_thresholds(timeframe, 1.0))
            
            if not validation_ok:
                self.logger.warning("⚠️ Validation finale échouée")
            else:
                self.logger.info("✓ Validation finale réussie")
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_merged_result: {str(e)}")

class DataProcessor(BaseDataProcessor):
    """Processeur générique pour les données financières"""
    
    def __init__(self, instrument_type='forex', is_base=False):
        """
        Args:
            instrument_type: Type d'instrument ('forex', 'gold', etc.)
            is_base: Si True, ne crée pas de processeur spécialisé
        """
        super().__init__()
        self.instrument_type = instrument_type
        self.processor = None if is_base else self._get_specialized_processor()
        
        # Configuration des indicateurs techniques
        self.technical_indicators_config = {
            'trend': {
                'EMA': {'periods': [12, 26, 50]},
                'MACD': {'fast': 12, 'slow': 26, 'signal': 9}
            }
        }
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Délègue le calcul des indicateurs à TechnicalIndicators"""
        if self.processor is not None:
            return self.processor.add_technical_indicators(df)
        
        try:
            self.logger.info("\nAjout des indicateurs techniques...")
            df = df.copy()
            
            # Vérification des données
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Colonnes manquantes: {missing}")
            
            # Utiliser TechnicalIndicators pour les calculs
            return self.indicator_calculator.calculate_all(df)
            
        except Exception as e:
            self.logger.error(f"Erreur dans add_technical_indicators: {str(e)}")
            return self._add_basic_indicators(df)

    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implémentation de la méthode abstraite"""
        if self.processor is not None:
            return self.processor.add_price_patterns(df)
        
        try:
            self.logger.info("\nDétection des patterns de prix...")
            return df  # Version de base : pas de patterns
            
        except Exception as e:
            self.logger.error(f"Erreur dans add_price_patterns: {str(e)}")
            return df
    
    def _get_specialized_processor(self):
        """Retourne le processeur spécialisé approprié"""
        if self.instrument_type.lower() == 'gold':
            return GoldDataProcessor(is_base=True)
        return None

class GoldDataProcessor(DataProcessor):
    """Processeur spécialisé pour l'or"""
    
    def __init__(self, is_base=False):
        super().__init__(instrument_type='gold', is_base=True)  # Toujours is_base=True
        
        # Configuration spécifique pour l'or
        self.technical_indicators_config.update({
            'gold_specific': {
                'Dollar_Index': {},
                'Gold_Volatility': {'period': 14},
                'Seasonal_Pattern': {}
            }
        })

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques avec gestion d'erreurs améliorée"""
        try:
            self.logger.info("\nAjout des indicateurs techniques...")
            df = df.copy()
            
            # Vérification préalable des données
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Colonnes manquantes: {missing}")
            
            # Configuration spécifique pour l'or
            gold_params = {
                'sma': [20, 50, 200],
                'ema': [12, 26],
                'rsi': [14],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bbands': {'period': 20, 'std_dev': 2},
                'stoch': {'k': 14, 'd': 3},
                'atr': [14],
                'adx': [14],
                'momentum': [10],
                'volume_ma': [20]
            }
            
            # Calcul des indicateurs avec TechnicalIndicators
            df = self.indicator_calculator.calculate_all(df, params=gold_params)
            
            # Log des statistiques
            self.logger.info("\nStatistiques des indicateurs:")
            for col in df.columns:
                if col not in required_cols:
                    stats = df[col].describe()
                    self.logger.info(f"\n{col}:")
                    self.logger.info(f"• Moyenne: {stats['mean']:.4f}")
                    self.logger.info(f"• Écart-type: {stats['std']:.4f}")
                    self.logger.info(f"• Min/Max: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur globale dans add_technical_indicators: {str(e)}")
            return self._add_basic_indicators(df)