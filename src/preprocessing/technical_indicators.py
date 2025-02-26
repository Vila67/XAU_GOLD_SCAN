import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import ta
import logging
import os
from datetime import datetime

class TechnicalIndicators:
    """Classe centralisée pour le calcul des indicateurs techniques"""
    
    def __init__(self, custom_config: Optional[Dict] = None):
        self.logger = self._setup_logging()
        
        # Configuration avancée des indicateurs
        self.base_config = {
            'trend': {
                'sma': {
                    'periods': [20, 50, 200],
                    'apply_to': 'Close',
                    'fill_method': 'ffill'
                },
                'ema': {
                    'periods': [12, 26],
                    'alpha': None,  # Utiliser la valeur par défaut
                    'adjust': True
                },
                'macd': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9,
                    'apply_to': 'Close'
                },
                'adx': {
                    'periods': [14],
                    'method': 'wilder',
                    'drift': 1
                },
                'ichimoku': {
                    'tenkan': 9,
                    'kijun': 26,
                    'senkou_span_b': 52,
                    'enable': False
                }
            },
            'momentum': {
                'rsi': {
                    'periods': [14],
                    'method': 'wilder',
                    'adjust': True
                },
                'stoch': {
                    'k_period': 14,
                    'd_period': 3,
                    'smooth_k': 3,
                    'method': 'sma'
                },
                'stoch_rsi': {
                    'period': 14,
                    'smooth1': 3,
                    'smooth2': 3,
                    'enable': True
                },
                'williams_r': {
                    'period': 14,
                    'enable': True
                }
            },
            'volatility': {
                'bbands': {
                    'period': 20,
                    'std_dev': 2,
                    'matype': 0,  # 0=SMA, 1=EMA, etc.
                    'bands': ['upper', 'middle', 'lower', 'width']
                },
                'atr': {
                    'periods': [14],
                    'normalize': True,
                    'adjust': True
                },
                'keltner': {
                    'period': 20,
                    'atr_period': 10,
                    'enable': False
                }
            },
            'volume': {
                'obv': {
                    'enable': True,
                    'normalize': True
                },
                'vwap': {
                    'enable': True,
                    'anchor': 'day'  # 'day', 'week', 'month'
                },
                'mfi': {
                    'periods': [14],
                    'normalize': True
                },
                'cmf': {
                    'period': 20,
                    'enable': True
                }
            },
            'custom': {
                'enable_all': False,
                'indicators': {}  # Pour les indicateurs personnalisés
            }
        }
        
        # Fusionner avec la configuration personnalisée
        self.config = self._merge_configs(self.base_config, custom_config or {})
        
        # Initialiser les indicateurs avec la nouvelle configuration
        self._init_indicators()

    def _merge_configs(self, base: Dict, custom: Dict) -> Dict:
        """Fusionne les configurations de manière récursive"""
        merged = base.copy()
        
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged

    def _init_indicators(self):
        """Initialise les indicateurs selon la configuration"""
        try:
            # Indicateurs de tendance
            self.trend_indicators = {
                'sma': lambda df, params: ta.trend.SMAIndicator(
                    close=df[params['apply_to']],
                    window=params['period']
                ).sma_indicator(),
                
                'ema': lambda df, params: ta.trend.EMAIndicator(
                    close=df[params['apply_to']],
                    window=params['period'],
                    alpha=params.get('alpha')
                ).ema_indicator(),
                
                'macd': lambda df, params: self._create_macd_indicator(df, params),
                
                'adx': lambda df, params: ta.trend.ADXIndicator(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    window=params['period']
                ).adx()
            }
            
            # Indicateurs de momentum personnalisables
            self.momentum_indicators = {
                'rsi': lambda df, params: ta.momentum.RSIIndicator(
                    close=df['Close'],
                    window=params['period'],
                    fillna=True
                ).rsi(),
                
                'stoch': lambda df, params: self._create_stoch_indicator(df, params),
                
                'williams_r': lambda df, params: ta.momentum.WilliamsRIndicator(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    lbp=params['period']
                ).williams_r()
            }
            
            # Ajouter d'autres catégories...
            
        except Exception as e:
            self.logger.error(f"Erreur dans _init_indicators: {str(e)}")

    def calculate_all(self, df: pd.DataFrame, instrument: str = 'XAUUSD',
                     custom_params: Optional[Dict] = None) -> pd.DataFrame:
        """Calcule les indicateurs avec paramètres personnalisables"""
        try:
            self.logger.info("\nCalcul des indicateurs techniques")
            df = df.copy()
            
            # Fusionner les paramètres spécifiques à l'instrument
            active_config = self.config.copy()
            if custom_params:
                active_config = self._merge_configs(active_config, custom_params)
            
            # Appliquer les indicateurs selon la configuration
            for category, indicators in active_config.items():
                if category == 'custom':
                    if indicators['enable_all']:
                        df = self._apply_custom_indicators(df, indicators['indicators'])
                    continue
                
                for ind_name, params in indicators.items():
                    if isinstance(params, dict) and params.get('enable', True):
                        df = self._apply_indicator(df, category, ind_name, params)
            
            return self._validate_and_clean(df)
            
        except Exception as e:
            self.logger.error(f"Erreur dans calculate_all: {str(e)}")
            return self._calculate_basic_indicators(df)

    def _apply_indicator(self, df: pd.DataFrame, category: str, 
                        indicator: str, params: Dict) -> pd.DataFrame:
        """Applique un indicateur avec ses paramètres"""
        try:
            indicator_fn = getattr(self, f"{category}_indicators").get(indicator)
            if indicator_fn is None:
                self.logger.warning(f"Indicateur non trouvé: {indicator}")
                return df
            
            # Gérer les périodes multiples
            if 'periods' in params:
                for period in params['periods']:
                    period_params = {**params, 'period': period}
                    df[f"{indicator.upper()}_{period}"] = indicator_fn(df, period_params)
            else:
                df[indicator.upper()] = indicator_fn(df, params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _apply_indicator: {str(e)}")
            return df

    def add_custom_indicator(self, name: str, function: callable, 
                           default_params: Dict) -> None:
        """Ajoute un indicateur personnalisé"""
        try:
            self.config['custom']['indicators'][name] = {
                'function': function,
                'params': default_params
            }
            self.logger.info(f"✓ Indicateur personnalisé ajouté: {name}")
            
        except Exception as e:
            self.logger.error(f"Erreur dans add_custom_indicator: {str(e)}")

    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcul optimisé des indicateurs de base"""
        try:
            # EMA optimisé
            indicator = self.trend_indicators['ema'](
                close=df['Close'],
                window=20
            )
            df['EMA_20'] = indicator.ema_indicator()
            
            # RSI optimisé
            indicator = self.momentum_indicators['rsi'](
                close=df['Close'],
                window=14
            )
            df['RSI_14'] = indicator.rsi()
            
            # ATR optimisé
            indicator = self.volatility_indicators['atr'](
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=14
            )
            df['ATR_14'] = indicator.average_true_range()
            
            self.logger.info("✓ Indicateurs de base calculés avec optimisation")
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _calculate_basic_indicators: {str(e)}")
            return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie les données avec une gestion intelligente des NaN"""
        try:
            # 1. Analyse des valeurs manquantes
            nan_stats = self._analyze_missing_values(df)
            
            # 2. Nettoyage par catégorie d'indicateur
            df = self._clean_by_category(df)
            
            # 3. Vérification finale
            remaining_nans = df.isna().sum()
            if remaining_nans.any():
                self.logger.warning("Valeurs manquantes restantes:")
                for col in remaining_nans[remaining_nans > 0].index:
                    self.logger.warning(f"• {col}: {remaining_nans[col]} NaN")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _validate_and_clean: {str(e)}")
            return df

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyse les valeurs manquantes pour choisir la meilleure stratégie"""
        try:
            nan_stats = {}
            
            for col in df.columns:
                nan_mask = df[col].isna()
                nan_count = nan_mask.sum()
                
                if nan_count > 0:
                    # Analyser la distribution des NaN
                    nan_sequences = self._get_nan_sequences(nan_mask)
                    
                    nan_stats[col] = {
                        'count': nan_count,
                        'percentage': (nan_count / len(df)) * 100,
                        'max_sequence': max(len(seq) for seq in nan_sequences),
                        'sequences': nan_sequences,
                        'location': 'start' if nan_mask.iloc[0] else 'end' if nan_mask.iloc[-1] else 'middle'
                    }
                    
                    # Log des statistiques
                    self.logger.info(f"\nAnalyse des NaN pour {col}:")
                    self.logger.info(f"• Nombre total: {nan_count}")
                    self.logger.info(f"• Pourcentage: {nan_stats[col]['percentage']:.2f}%")
                    self.logger.info(f"• Séquence max: {nan_stats[col]['max_sequence']}")
                    self.logger.info(f"• Position: {nan_stats[col]['location']}")
            
            return nan_stats
            
        except Exception as e:
            self.logger.error(f"Erreur dans _analyze_missing_values: {str(e)}")
            return {}

    def _get_nan_sequences(self, nan_mask: pd.Series) -> List[pd.Series]:
        """Identifie les séquences de valeurs manquantes"""
        sequences = []
        current_sequence = []
        
        for i, is_nan in enumerate(nan_mask):
            if is_nan:
                current_sequence.append(i)
            elif current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
        
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences

    def _clean_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données selon le type d'indicateur"""
        try:
            # 1. Nettoyage des indicateurs de tendance
            trend_cols = [col for col in df.columns if any(
                ind in col for ind in ['SMA', 'EMA', 'MACD', 'ADX']
            )]
            df = self._clean_trend_indicators(df, trend_cols)
            
            # 2. Nettoyage des indicateurs de momentum
            momentum_cols = [col for col in df.columns if any(
                ind in col for ind in ['RSI', 'STOCH', 'CCI', 'MOM']
            )]
            df = self._clean_momentum_indicators(df, momentum_cols)
            
            # 3. Nettoyage des indicateurs de volatilité
            volatility_cols = [col for col in df.columns if any(
                ind in col for ind in ['BB', 'ATR', 'NATR']
            )]
            df = self._clean_volatility_indicators(df, volatility_cols)
            
            # 4. Nettoyage des indicateurs de volume
            volume_cols = [col for col in df.columns if any(
                ind in col for ind in ['OBV', 'VWAP', 'MFI']
            )]
            df = self._clean_volume_indicators(df, volume_cols)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _clean_by_category: {str(e)}")
            return df

    def _clean_trend_indicators(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Nettoie les indicateurs de tendance"""
        try:
            for col in cols:
                nan_mask = df[col].isna()
                if not nan_mask.any():
                    continue
                
                # Interpolation linéaire pour les tendances
                if nan_mask.sum() / len(df) < 0.1:  # < 10% de NaN
                    df[col] = df[col].interpolate(method='linear')
                else:
                    # Pour les grandes séquences, utiliser une approche plus sophistiquée
                    df[col] = self._advanced_interpolation(df[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _clean_trend_indicators: {str(e)}")
            return df

    def _clean_momentum_indicators(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Nettoie les indicateurs de momentum"""
        try:
            for col in cols:
                # Les indicateurs de momentum nécessitent une approche spéciale
                if 'RSI' in col:
                    # RSI doit rester entre 0 et 100
                    df[col] = df[col].fillna(50)  # Valeur neutre
                elif 'STOCH' in col:
                    # Stochastique aussi entre 0 et 100
                    df[col] = df[col].interpolate(method='linear').clip(0, 100)
                else:
                    # Autres indicateurs de momentum
                    df[col] = df[col].interpolate(method='akima', limit=5)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _clean_momentum_indicators: {str(e)}")
            return df

    def _clean_volatility_indicators(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Nettoie les indicateurs de volatilité"""
        try:
            for col in cols:
                if 'ATR' in col or 'NATR' in col:
                    # ATR ne peut pas être négatif
                    df[col] = df[col].interpolate(method='linear').clip(lower=0)
                elif 'BB' in col:
                    # Bandes de Bollinger
                    if 'Upper' in col or 'Lower' in col:
                        df[col] = df[col].interpolate(method='cubic', limit=3)
                    else:
                        df[col] = df[col].interpolate(method='linear')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur dans _clean_volatility_indicators: {str(e)}")
            return df

    def _advanced_interpolation(self, series: pd.Series) -> pd.Series:
        """Interpolation avancée pour les grandes séquences de NaN"""
        try:
            # 1. Identifier les séquences de NaN
            nan_sequences = self._get_nan_sequences(series.isna())
            
            # 2. Traiter chaque séquence
            for sequence in nan_sequences:
                seq_length = len(sequence)
                
                if seq_length <= 3:
                    # Petites séquences : interpolation linéaire
                    series.iloc[sequence] = series.interpolate(method='linear').iloc[sequence]
                elif seq_length <= 10:
                    # Séquences moyennes : interpolation cubique
                    series.iloc[sequence] = series.interpolate(method='cubic').iloc[sequence]
                else:
                    # Grandes séquences : combinaison de méthodes
                    start_val = series.iloc[sequence[0]-1] if sequence[0] > 0 else series.mean()
                    end_val = series.iloc[sequence[-1]+1] if sequence[-1] < len(series)-1 else series.mean()
                    
                    # Créer une tendance entre start_val et end_val
                    trend = np.linspace(start_val, end_val, seq_length)
                    
                    # Ajouter une composante cyclique si nécessaire
                    if 'cyclic' in series.name.lower():
                        cycle = np.sin(np.linspace(0, 2*np.pi, seq_length)) * series.std() * 0.1
                        trend += cycle
                    
                    series.iloc[sequence] = trend
            
            return series
            
        except Exception as e:
            self.logger.error(f"Erreur dans _advanced_interpolation: {str(e)}")
            return series

    def _log_indicator_stats(self, df: pd.DataFrame):
        """Log les statistiques des indicateurs"""
        try:
            self.logger.info("\nStatistiques des indicateurs:")
            
            # Grouper par type d'indicateur
            indicator_groups = {
                'Trend': ['SMA', 'EMA', 'MACD', 'ADX'],
                'Momentum': ['RSI', 'STOCH', 'CCI', 'MOM'],
                'Volatility': ['BB', 'ATR', 'NATR'],
                'Volume': ['OBV', 'Volume_MA', 'VWAP', 'MFI']
            }
            
            for group, prefixes in indicator_groups.items():
                group_cols = [
                    col for col in df.columns 
                    if any(col.startswith(prefix) for prefix in prefixes)
                ]
                
                if group_cols:
                    self.logger.info(f"\n{group}:")
                    for col in group_cols:
                        stats = df[col].describe()
                        self.logger.info(f"\n{col}:")
                        self.logger.info(f"• Moyenne: {stats['mean']:.4f}")
                        self.logger.info(f"• Écart-type: {stats['std']:.4f}")
                        self.logger.info(f"• Min/Max: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
        except Exception as e:
            self.logger.error(f"Erreur dans _log_indicator_stats: {str(e)}")

    def _setup_logging(self):
        """Configure le système de logging"""
        logger = logging.getLogger('TechnicalIndicators')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Handler pour fichier
            os.makedirs('logs/indicators', exist_ok=True)
            fh = logging.FileHandler(
                f'logs/indicators/indicators_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
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