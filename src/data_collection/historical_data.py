import os
import pandas as pd
import chardet
import csv
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
from datetime import datetime
import yaml
from typing import Dict, Optional, Tuple, List
import numpy as np
import requests
from ratelimit import limits, sleep_and_retry
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from kaggle.rest import ApiException
import json
import shutil
import asyncio

class Config:
    def __init__(self):
        self.config = {
            'data_encoding': 'utf-8',
            'data_separator': ';',
            # Autres configurations par défaut
        }
    
    def get(self, key: str, default=None):
        """
        Récupère une valeur de configuration
        
        Args:
            key: Clé de configuration
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """
        Définit une valeur de configuration
        
        Args:
            key: Clé de configuration
            value: Valeur à définir
        """
        self.config[key] = value

class HistoricalDataLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_path = os.path.join('data')
        os.makedirs(self.base_path, exist_ok=True)
        
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Initialiser les mappings depuis la configuration
        self.timeframes = self.config['timeframes']
        self.timeframe_mapping = self.config['mappings']['default']
        
        # Validation de la configuration
        self._validate_config()

        # Configuration pour le live trading
        self.realtime_config = {
            'marketstack': {
                'base_url': 'http://api.marketstack.com/v1',
                'api_key_env': 'MARKETSTACK_API_KEY',
                'rate_limit': 100,  # Requêtes par minute
                'symbols': {
                    'XAUUSD': 'XAU/USD'  # Mapping des symboles
                },
                'timeframes': {
                    '5m': '5min',
                    '15m': '15min',
                    '30m': '30min',
                    '1h': '1hour',
                    '4h': '4hour',
                    '1d': 'day'
                }
            }
        }

        # Configuration pour le live trading
        self.live_config = {
            'sequence_length': 60,  # Longueur des séquences pour le ML
            'update_interval': 60,  # Secondes entre les mises à jour
            'buffer_size': 1000,    # Taille du buffer de données historiques
            'warmup_periods': {     # Périodes de warmup par timeframe
                '5m': 300,
                '15m': 100,
                '30m': 50,
                '1h': 24,
                '4h': 10
            }
        }
        
        # Buffers pour les données live
        self.live_buffers = {}
        self.last_updates = {}

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration depuis un fichier YAML ou utilise les valeurs par défaut"""
        default_config = {
            'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'],
            'mappings': {
                'default': {
                    '5m': 'XAU_5m_data.csv',
                    '15m': 'XAU_15m_data.csv',
                    '30m': 'XAU_30m_data.csv',
                    '1h': 'XAU_1h_data.csv',
                    '4h': 'XAU_4h_data.csv',
                    '1d': 'XAU_1d_data.csv',
                    '1w': 'XAU_1w_data.csv',
                    '1M': 'XAU_1Month_data.csv'
                },
                'kaggle': {
                    '5m': 'XAUUSD_5Minutes.csv',
                    '15m': 'XAUUSD_15Minutes.csv',
                    '30m': 'XAUUSD_30Minutes.csv',
                    '1h': 'XAUUSD_H1.csv',
                    '4h': 'XAUUSD_H4.csv',
                    '1d': 'XAUUSD_D1.csv',
                    '1w': 'XAUUSD_W1.csv',
                    '1M': 'XAUUSD_MN.csv'
                },
                'mt5': {
                    '5m': 'XAUUSD5.csv',
                    '15m': 'XAUUSD15.csv',
                    '30m': 'XAUUSD30.csv',
                    '1h': 'XAUUSD60.csv',
                    '4h': 'XAUUSD240.csv',
                    '1d': 'XAUUSDD1.csv',
                    '1w': 'XAUUSDW1.csv',
                    '1M': 'XAUUSDMN1.csv'
                }
            },
            'instruments': {
                'XAUUSD': {
                    'description': 'Or vs USD',
                    'pip_value': 0.01,
                    'min_price': 1000.0,
                    'max_price': 3000.0,
                    'decimal_places': 2
                }
            },
            'sources': {
                'kaggle': {
                    'enabled': True,
                    'dataset': 'novandraanugrah/xauusd-gold-price-historical-data-2004-2024',
                    'api_key_env': 'KAGGLE_API_KEY'
                },
                'mt5': {
                    'enabled': False,
                    'server': 'ICMarkets-Demo',
                    'login': 12345678,
                    'password_env': 'MT5_PASSWORD'
                }
            }
        }
        
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    # Fusion récursive des configurations
                    return self._merge_configs(default_config, custom_config)
            
            return default_config
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return default_config

    def _merge_configs(self, base: Dict, custom: Dict) -> Dict:
        """Fusionne récursivement deux dictionnaires de configuration"""
        merged = base.copy()
        
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged

    def _validate_config(self):
        """Valide la configuration chargée"""
        try:
            # Vérifier les timeframes
            for tf in self.timeframes:
                if tf not in self.timeframe_mapping:
                    self.logger.warning(f"Mapping manquant pour le timeframe {tf}")
            
            # Vérifier les sources de données
            for source, config in self.config['sources'].items():
                if config['enabled']:
                    self._validate_source_config(source, config)
            
            # Vérifier les instruments
            for instrument, config in self.config['instruments'].items():
                required_fields = ['pip_value', 'min_price', 'max_price']
                missing = [f for f in required_fields if f not in config]
                if missing:
                    self.logger.warning(f"Configuration incomplète pour {instrument}: {missing}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation de la configuration: {str(e)}")

    def _validate_source_config(self, source: str, config: Dict):
        """Valide la configuration d'une source de données"""
        try:
            if source == 'kaggle':
                api_key = os.getenv(config['api_key_env'])
                if not api_key:
                    self.logger.warning(f"Clé API Kaggle manquante ({config['api_key_env']})")
            
            elif source == 'mt5':
                password = os.getenv(config['password_env'])
                if not password:
                    self.logger.warning(f"Mot de passe MT5 manquant ({config['password_env']})")
            
        except Exception as e:
            self.logger.error(f"Erreur validation source {source}: {str(e)}")

    def set_data_source(self, source: str):
        """Change la source de données active"""
        try:
            if source not in self.config['mappings']:
                raise ValueError(f"Source non supportée: {source}")
            
            if not self.config['sources'].get(source, {}).get('enabled', False):
                raise ValueError(f"Source {source} désactivée dans la configuration")
            
            self.timeframe_mapping = self.config['mappings'][source]
            self.logger.info(f"Source de données changée pour: {source}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du changement de source: {str(e)}")

    def add_custom_mapping(self, timeframe: str, filename: str, source: str = 'default'):
        """Ajoute un mapping personnalisé"""
        try:
            if source not in self.config['mappings']:
                self.config['mappings'][source] = {}
            
            self.config['mappings'][source][timeframe] = filename
            
            if source == 'default':
                self.timeframe_mapping[timeframe] = filename
            
            self.logger.info(f"Mapping ajouté: {timeframe} -> {filename} (source: {source})")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout du mapping: {str(e)}")

    def save_config(self, path: str):
        """Sauvegarde la configuration actuelle"""
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration sauvegardée: {path}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")

    def _get_data_path(self, timeframe):
        """Retourne le chemin du fichier de données pour un timeframe donné"""
        if timeframe not in self.timeframe_mapping:
            self.logger.error(f"Timeframe invalide: {timeframe}")
            raise ValueError(f"Timeframe invalide: {timeframe}")
        
        filename = self.timeframe_mapping[timeframe]
        return os.path.join(self.base_path, filename)

    def load_data(self, timeframe):
        """Charge les données historiques depuis un fichier CSV"""
        try:
            file_path = self._get_data_path(timeframe)
            if not os.path.exists(file_path):
                self.logger.error(f"Fichier non trouvé: {file_path}")
                return None

            # Lecture du CSV avec gestion plus flexible du séparateur
            try:
                # Essayer d'abord avec point-virgule
                df = pd.read_csv(file_path, sep=';', encoding='utf-8', skipinitialspace=True)
            except:
                # Si échec, essayer avec virgule
                df = pd.read_csv(file_path, sep=',', encoding='utf-8', skipinitialspace=True)
            
            # Nettoyer les noms de colonnes (garder la majuscule initiale)
            df.columns = [col.strip().title() for col in df.columns]
            
            self.logger.info(f"Colonnes trouvées dans {timeframe}: {list(df.columns)}")
            
            # Vérification des colonnes requises
            required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                self.logger.error(f"Colonnes manquantes dans {timeframe}: {missing_cols}")
                return None

            # Nettoyage des données
            df = df.dropna(subset=['Date'])
            
            # Conversion de la colonne Date avec gestion d'erreurs détaillée
            try:
                # Vérifier le format des dates
                sample_date = df['Date'].iloc[0]
                self.logger.info(f"Format de date détecté: {sample_date}")
                
                # Essayer différents formats de date
                date_formats = [
                    '%Y.%m.%d %H:%M',
                    '%Y-%m-%d %H:%M',
                    '%d.%m.%Y %H:%M',
                    '%Y.%m.%d %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S'
                ]
                
                for date_format in date_formats:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
                        self.logger.info(f"Conversion réussie avec format: {date_format}")
                        break
                    except ValueError:
                        continue
                else:
                    # Si aucun format ne fonctionne, essayer la détection automatique
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Vérifier si la conversion a réussi
                if df['Date'].isna().any():
                    problematic_dates = df[df['Date'].isna()]['Date'].head()
                    self.logger.error(f"Dates problématiques: {problematic_dates.tolist()}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Erreur conversion date pour {timeframe}: {str(e)}")
                self.logger.info(f"Exemple de valeurs Date:\n{df['Date'].head()}")
                return None

            # Convertir les colonnes numériques avec gestion d'erreurs
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                try:
                    # Remplacer les virgules par des points si nécessaire
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    self.logger.error(f"Erreur conversion {col}: {str(e)}")
                    self.logger.info(f"Exemple de valeurs {col}:\n{df[col].head()}")
                    return None

            # Supprimer les lignes avec des valeurs manquantes
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                self.logger.warning(f"Lignes supprimées: {dropped_rows} ({dropped_rows/initial_rows:.1%})")

            # Définir Date comme index et vérifier
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("Index temporel non défini correctement")
                return None

            # Tri chronologique
            df.sort_index(inplace=True)

            # Vérification finale de l'index
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("L'index n'est pas un DatetimeIndex après traitement")
                return None

            self.logger.info(f"✓ Données {timeframe} chargées: {len(df):,} lignes")
            return df

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {timeframe}: {str(e)}")
            # Afficher le début du fichier pour diagnostic
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.logger.info(f"Aperçu du fichier {timeframe}:\n{f.read(500)}")
            except Exception as read_error:
                self.logger.error(f"Impossible de lire le fichier: {str(read_error)}")
            return None

    def _setup_logging(self):
        """Configure le système de logging"""
        log_dir = 'logs/data_collection'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('HistoricalDataLoader')
        self.logger.setLevel(logging.DEBUG)
        
        # Handler pour fichier avec encodage UTF-8
        fh = logging.FileHandler(
            f'{log_dir}/data_loading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        
        # Handler pour console avec encodage UTF-8
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter sans caractères spéciaux
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _detect_file_encoding(self, file_path):
        """
        Détecte l'encodage d'un fichier
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            str: Encodage détecté
        """
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.info(f"Encodage détecté pour {file_path}:")
                self.logger.info(f"• Encodage: {encoding}")
                self.logger.info(f"• Confiance: {confidence:.2%}")
                
                return encoding if confidence > 0.7 else self.config.get('data_encoding', 'utf-8')
        except Exception as e:
            self.logger.warning(f"Erreur lors de la détection de l'encodage: {str(e)}")
            return self.config.get('data_encoding', 'utf-8')
    
    def _detect_separator(self, file_path, encoding):
        """
        Détecte le séparateur d'un fichier CSV
        
        Args:
            file_path: Chemin du fichier
            encoding: Encodage du fichier
            
        Returns:
            str: Séparateur détecté
        """
        try:
            # Lire les premières lignes
            with open(file_path, 'r', encoding=encoding) as file:
                sample = ''.join(file.readline() for _ in range(5))
            
            # Tester différents séparateurs
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            separator = dialect.delimiter
            
            self.logger.info(f"Séparateur détecté pour {file_path}: '{separator}'")
            return separator
        except Exception as e:
            self.logger.warning(f"Erreur lors de la détection du séparateur: {str(e)}")
            return self.config.get('data_separator', ';')
    
    def _validate_temporal_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """Valide la cohérence temporelle des données de manière optimisée"""
        try:
            corrections_made = False
            chunk_size = 100000  # Taille de chunk optimale
            
            # Configuration des intervalles attendus
            expected_intervals = {
                '5m': pd.Timedelta(minutes=5),
                '15m': pd.Timedelta(minutes=15),
                '30m': pd.Timedelta(minutes=30),
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                '1d': pd.Timedelta(days=1),
                '1w': pd.Timedelta(weeks=1),
                '1M': pd.Timedelta(days=30)
            }
            interval = expected_intervals[timeframe]
            
            # 1. Traitement des doublons de manière vectorisée
            duplicates = df.index.duplicated(keep='first')
            if duplicates.any():
                n_duplicates = duplicates.sum()
                self.logger.warning(f"Doublons détectés dans {timeframe}: {n_duplicates} entrées")
                df = df[~duplicates]
                corrections_made = True
            
            # 2. Vérification des gaps par chunks
            def process_gaps_chunk(chunk_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
                date_diff = chunk_df.index.to_series().diff()
                gaps = date_diff > interval * 1.5
                chunk_corrections = False
                
                if gaps.any():
                    gap_indices = gaps[gaps].index
                    gap_sizes = date_diff[gap_indices]
                    reasonable_gaps = gap_sizes <= (interval * 5)
                    
                    if reasonable_gaps.any():
                        # Traitement vectorisé des gaps raisonnables
                        new_indices = []
                        new_values = []
                        
                        for idx in gap_indices[reasonable_gaps]:
                            gap_start = chunk_df.index[chunk_df.index.get_loc(idx)-1]
                            gap_end = idx
                            
                            # Créer des points intermédiaires
                            new_points = pd.date_range(
                                start=gap_start,
                                end=gap_end,
                                freq=interval
                            )[1:-1]
                            
                            if len(new_points) > 0:
                                # Interpolation vectorisée
                                start_values = chunk_df.loc[gap_start].values
                                end_values = chunk_df.loc[gap_end].values
                                
                                # Générer les valeurs interpolées
                                steps = np.linspace(0, 1, len(new_points) + 2)[1:-1]
                                for i, point in enumerate(new_points):
                                    new_indices.append(point)
                                    new_values.append(
                                        start_values + (end_values - start_values) * steps[i]
                                    )
                        
                        if new_indices:
                            # Ajouter les points interpolés
                            new_df = pd.DataFrame(
                                new_values,
                                index=new_indices,
                                columns=chunk_df.columns
                            )
                            chunk_df = pd.concat([chunk_df, new_df]).sort_index()
                            chunk_corrections = True
                
                return chunk_df, chunk_corrections
            
            # Traitement par chunks
            chunks = []
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                chunk = df.iloc[start:end].copy()
                processed_chunk, chunk_corrected = process_gaps_chunk(chunk)
                chunks.append(processed_chunk)
                corrections_made |= chunk_corrected
            
            # Reconstituer le DataFrame
            df = pd.concat(chunks).sort_index()
            
            # 3. Vérification des valeurs aberrantes de manière vectorisée
            def process_outliers_vectorized(series: pd.Series) -> Tuple[pd.Series, bool]:
                zscore = (series - series.mean()) / series.std()
                outliers = abs(zscore) > 3
                
                if outliers.any():
                    # Correction vectorisée avec rolling window
                    series.loc[outliers] = series.rolling(
                        window=5, center=True, min_periods=1
                    ).mean().loc[outliers]
                    return series, True
                return series, False
            
            # Traitement vectorisé des colonnes OHLC
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col], col_corrected = process_outliers_vectorized(df[col])
                corrections_made |= col_corrected
            
            # 4. Vérification de la cohérence OHLC de manière vectorisée
            invalid_high = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close'])
            invalid_low = (df['Low'] > df['High']) | (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
            
            if invalid_high.any() or invalid_low.any():
                # Correction vectorisée
                affected_rows = invalid_high | invalid_low
                df.loc[affected_rows, 'High'] = df.loc[affected_rows, ['Open', 'High', 'Low', 'Close']].max(axis=1)
                df.loc[affected_rows, 'Low'] = df.loc[affected_rows, ['Open', 'High', 'Low', 'Close']].min(axis=1)
                corrections_made = True
            
            # 5. Traitement du volume de manière vectorisée
            zero_volume = df['Volume'] == 0
            if zero_volume.any():
                # Correction vectorisée avec rolling window
                df.loc[zero_volume, 'Volume'] = df['Volume'].rolling(
                    window=5, center=True, min_periods=1
                ).mean().loc[zero_volume]
                corrections_made = True
            
            # 6. Statistiques finales
            if corrections_made:
                self.logger.info("\nStatistiques après corrections:")
                self.logger.info(f"• Points totaux: {len(df):,}")
                self.logger.info(f"• Intervalle moyen: {df.index.to_series().diff().mean()}")
                
                # Vérification de la qualité
                remaining_gaps = (df.index.to_series().diff() > interval * 1.5).sum()
                if remaining_gaps > 0:
                    self.logger.warning(f"Gaps restants: {remaining_gaps}")
                
                # Statistiques des prix
                for col in ['Open', 'High', 'Low', 'Close']:
                    stats = df[col].describe()
                    self.logger.info(f"\n{col}:")
                    self.logger.info(f"• Moyenne: {stats['mean']:.2f}")
                    self.logger.info(f"• Écart-type: {stats['std']:.2f}")
                    self.logger.info(f"• Min/Max: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            return df, corrections_made
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation temporelle: {str(e)}")
            return df, False
    
    def load_multi_timeframe_data(self):
        """Charge les données pour tous les timeframes avec validation"""
        self.logger.info("\nDébut du chargement des données multi-timeframe")
        data_dict = {}
        
        # Vérifier les fichiers disponibles avant de commencer
        available_files = [tf for tf in self.timeframes 
                          if os.path.exists(self._get_data_path(tf))]
        
        if not available_files:
            self.logger.warning("\n⚠️ Aucun fichier de données trouvé")
            self.logger.info("Tentative de téléchargement depuis Kaggle...")
            for tf in self.timeframes:
                try:
                    self._download_data(tf)
                except Exception as e:
                    self.logger.error(f"Échec du téléchargement pour {tf}: {str(e)}")
        
        for tf in self.timeframes:
            try:
                # Utiliser load_data pour charger les données
                df = self.load_data(timeframe=tf)
                
                if df is None or df.empty:
                    self.logger.warning(f"\n⚠️ Pas de données pour {tf}")
                    continue

                # Correction des prix si nécessaire
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if df[col].mean() < 1000:
                        self.logger.warning(f"Prix {col} trop bas, application d'un multiplicateur")
                        df[col] = df[col] * 1000
                    df[col] = df[col].round(4)
                
                self.logger.info(f"\nDonnées {tf} chargées:")
                self.logger.info(f"• Lignes: {len(df):,}")
                self.logger.info(f"• Période: {df.index.min()} à {df.index.max()}")
                
                # Statistiques des prix
                self.logger.info("\nStatistiques des prix:")
                self.logger.info(f"• Moyen: {df['Close'].mean():.2f}")
                self.logger.info(f"• Min: {df['Close'].min():.2f}")
                self.logger.info(f"• Max: {df['Close'].max():.2f}")
                
                # Validation temporelle
                df, corrections_made = self._validate_temporal_data(df, tf)
                
                if corrections_made:
                    self.logger.warning(f"\nDes corrections ont été appliquées aux données {tf}")
                    self.logger.info("Après corrections:")
                    self.logger.info(f"• Lignes: {len(df):,}")
                    self.logger.info(f"• Intervalle moyen: {df.index.to_series().diff().mean()}")
                
                data_dict[tf] = df
                self.logger.info(f"✓ {tf} chargé avec succès")

            except Exception as e:
                self.logger.error(f"\n❌ Erreur lors du chargement de {tf}:")
                self.logger.error(f"• Type: {type(e).__name__}")
                self.logger.error(f"• Message: {str(e)}")
                continue
        
        # Validation finale
        if not data_dict:
            raise ValueError("Aucun fichier n'a pu être chargé correctement")
        
        try:
            data_dict = self._validate_timeframe_consistency(data_dict)
            self.logger.info("\n✓ Validation inter-timeframes terminée avec succès")
            
            # Résumé final
            self.logger.info("\nRésumé du chargement:")
            for tf in self.timeframes:
                status = "✓" if tf in data_dict else "❌"
                self.logger.info(f"• {status} {tf}")
            
        except Exception as e:
            self.logger.error(f"\n❌ Erreur lors de la validation inter-timeframes:")
            self.logger.error(f"• Type: {type(e).__name__}")
            self.logger.error(f"• Message: {str(e)}")
        
        return data_dict
    
    def _download_data(self, timeframe):
        """Télécharge les données depuis Kaggle avec gestion améliorée de l'authentification"""
        try:
            # Vérification et configuration explicite des credentials Kaggle
            kaggle_dir = os.path.expanduser('~/.kaggle')
            kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
            
            if not os.path.exists(kaggle_json):
                raise ValueError(f"Fichier kaggle.json non trouvé dans {kaggle_dir}")
            
            # Vérifier les permissions du fichier kaggle.json
            import stat
            current_permissions = os.stat(kaggle_json).st_mode
            if (current_permissions & stat.S_IROTH) or (current_permissions & stat.S_IWOTH):
                # Corriger les permissions si nécessaire (600)
                os.chmod(kaggle_json, stat.S_IRUSR | stat.S_IWUSR)
                self.logger.info("Permissions de kaggle.json corrigées")
            
            # Charger et vérifier le contenu du fichier kaggle.json
            try:
                with open(kaggle_json, 'r') as f:
                    credentials = json.load(f)
                    if 'username' not in credentials or 'key' not in credentials:
                        raise ValueError("Format de kaggle.json invalide")
                    
                    # Définir explicitement les variables d'environnement
                    os.environ['KAGGLE_USERNAME'] = credentials['username']
                    os.environ['KAGGLE_KEY'] = credentials['key']
                    
                    self.logger.info("✓ Credentials Kaggle chargés avec succès")
            except Exception as e:
                raise ValueError(f"Erreur lecture kaggle.json: {str(e)}")
            
            # Configuration Kaggle
            dataset = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
            target_file = self.timeframe_mapping[timeframe]
            
            # Initialiser l'API Kaggle
            try:
                api = KaggleApi()
                api.authenticate()
                self.logger.info("✓ Authentification Kaggle réussie")
            except Exception as e:
                raise ValueError(f"Erreur authentification Kaggle: {str(e)}")
            
            # Téléchargement des données
            try:
                target_path = self._get_data_path(timeframe)
                self.logger.info(f"\nTéléchargement {timeframe}:")
                self.logger.info(f"• Source: {dataset}")
                self.logger.info(f"• Fichier: {target_file}")
                self.logger.info(f"• Destination: {target_path}")
                
                # Tentative de téléchargement direct
                try:
                    api.dataset_download_file(
                        dataset,
                        file_name=target_file,
                        path=self.base_path,
                        quiet=False
                    )
                    self.logger.info("✓ Téléchargement direct réussi")
                    
                except Exception as e:
                    self.logger.warning(f"\n⚠️ Erreur téléchargement direct: {str(e)}")
                    self.logger.info("Tentative de solution alternative...")
                    
                    # Solution alternative : télécharger tout le dataset
                    temp_dir = os.path.join(self.base_path, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    api.dataset_download_files(
                        dataset,
                        path=temp_dir,
                        unzip=True,
                        quiet=False
                    )
                    
                    # Chercher le fichier dans le dossier temporaire
                    source_file = os.path.join(temp_dir, target_file)
                    if os.path.exists(source_file):
                        import shutil
                        shutil.copy2(source_file, target_path)
                        self.logger.info("✓ Fichier récupéré via téléchargement complet")
                    else:
                        raise FileNotFoundError(f"Fichier {target_file} introuvable dans le dataset")
                    
                    # Nettoyer le dossier temporaire
                    shutil.rmtree(temp_dir, ignore_errors=True)
                
                return True
                
            except Exception as e:
                self.logger.error(f"\n❌ Erreur lors du téléchargement:")
                self.logger.error(f"• Type: {type(e).__name__}")
                self.logger.error(f"• Message: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Erreur dans _download_data: {str(e)}")
            raise

    def _validate_timeframe_consistency(self, data_dict):
        """Valide la cohérence des données entre les différents timeframes"""
        try:
            if not data_dict:
                self.logger.error("Aucune donnée à valider")
                return data_dict

            # Utiliser l'index au lieu de la colonne 'Date'
            for tf1, df1 in data_dict.items():
                for tf2, df2 in data_dict.items():
                    if tf1 != tf2:
                        # Convertir les index en timestamps pour la comparaison
                        dates1 = df1.index.astype(np.int64) // 10**9
                        dates2 = df2.index.astype(np.int64) // 10**9
                        
                        # Vérifier le chevauchement des périodes
                        start_overlap = max(dates1.min(), dates2.min())
                        end_overlap = min(dates1.max(), dates2.max())
                        
                        if start_overlap >= end_overlap:
                            self.logger.error(
                                f"Pas de chevauchement temporel entre {tf1} et {tf2}"
                            )
                            return data_dict
                        
                        # Vérifier la cohérence des données
                        overlap_period = (end_overlap - start_overlap) / (24 * 3600)  # en jours
                        if overlap_period < 30:  # Minimum 30 jours de chevauchement
                            self.logger.error(
                                f"Période de chevauchement insuffisante entre {tf1} et {tf2}: "
                                f"{overlap_period:.1f} jours"
                            )
                            return data_dict
            
            self.logger.info("✅ Validation inter-timeframes réussie")
            return data_dict
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la validation inter-timeframes: {str(e)}")
            return data_dict

    @sleep_and_retry
    @limits(calls=100, period=60)  # Rate limiting: 100 appels par minute
    def fetch_realtime_data(self, symbol: str, timeframe: str, 
                           limit: int = 100) -> pd.DataFrame:
        """
        Récupère les données en temps réel depuis Marketstack
        
        Args:
            symbol: Symbole de l'instrument (ex: 'XAUUSD')
            timeframe: Timeframe désiré ('5m', '15m', etc.)
            limit: Nombre de barres à récupérer
            
        Returns:
            DataFrame avec les données en temps réel
        """
        try:
            # 1. Validation des paramètres
            if symbol not in self.realtime_config['marketstack']['symbols']:
                raise ValueError(f"Symbole non supporté: {symbol}")
            
            if timeframe not in self.realtime_config['marketstack']['timeframes']:
                raise ValueError(f"Timeframe non supporté: {timeframe}")
            
            # 2. Préparation de la requête
            api_key = os.getenv(self.realtime_config['marketstack']['api_key_env'])
            if not api_key:
                raise ValueError("Clé API Marketstack manquante")
            
            base_url = self.realtime_config['marketstack']['base_url']
            ms_symbol = self.realtime_config['marketstack']['symbols'][symbol]
            ms_interval = self.realtime_config['marketstack']['timeframes'][timeframe]
            
            # 3. Construction de l'URL
            endpoint = f"{base_url}/intraday"
            params = {
                'access_key': api_key,
                'symbols': ms_symbol,
                'interval': ms_interval,
                'limit': limit,
                'sort': 'desc'
            }
            
            # 4. Exécution de la requête
            self.logger.info(f"\nRécupération des données temps réel:")
            self.logger.info(f"• Symbole: {symbol}")
            self.logger.info(f"• Timeframe: {timeframe}")
            self.logger.info(f"• Limite: {limit} barres")
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            # 5. Parsing de la réponse
            data = response.json()
            
            if 'data' not in data or not data['data']:
                raise ValueError("Aucune donnée reçue de l'API")
            
            # 6. Conversion en DataFrame
            df = self._parse_marketstack_response(data['data'])
            
            # 7. Validation et nettoyage
            df, corrections = self._validate_temporal_data(df, timeframe)
            
            if corrections:
                self.logger.warning("Des corrections ont été appliquées aux données temps réel")
            
            # 8. Statistiques
            self.logger.info("\nDonnées reçues:")
            self.logger.info(f"• Nombre de barres: {len(df)}")
            self.logger.info(f"• Dernière mise à jour: {df.index[0]}")
            self.logger.info(f"• Première barre: {df.index[-1]}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erreur de requête API: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {str(e)}")
            raise

    def _parse_marketstack_response(self, data: List[Dict]) -> pd.DataFrame:
        """Parse la réponse de l'API Marketstack"""
        try:
            # Conversion des données
            parsed_data = []
            for item in data:
                parsed_item = {
                    'Date': pd.to_datetime(item['date']),
                    'Open': float(item['open']),
                    'High': float(item['high']),
                    'Low': float(item['low']),
                    'Close': float(item['close']),
                    'Volume': int(item['volume'])
                }
                parsed_data.append(parsed_item)
            
            # Création du DataFrame
            df = pd.DataFrame(parsed_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors du parsing des données: {str(e)}")
            raise

    def stream_realtime_data(self, symbol: str, timeframe: str, 
                           callback: callable, interval: int = 60):
        """
        Stream continu des données en temps réel
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Timeframe désiré
            callback: Fonction à appeler avec les nouvelles données
            interval: Intervalle de mise à jour en secondes
        """
        try:
            while True:
                # Récupérer les données
                df = self.fetch_realtime_data(symbol, timeframe, limit=1)
                
                if not df.empty:
                    # Appeler le callback avec les nouvelles données
                    callback(df)
                
                # Attendre l'intervalle configuré
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Stream interrompu par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur dans le stream: {str(e)}")
            raise

    async def live_predict(self, model, timeframes: List[str], 
                         callback: callable = None) -> None:
        """
        Trading en temps réel avec prédictions continues
        
        Args:
            model: Modèle ML pour les prédictions
            timeframes: Liste des timeframes à surveiller
            callback: Fonction à appeler avec les prédictions
        """
        try:
            # 1. Initialisation des buffers
            await self._initialize_live_buffers(timeframes)
            
            self.logger.info("\n🚀 Démarrage du trading en temps réel")
            self.logger.info(f"• Timeframes surveillés: {timeframes}")
            self.logger.info(f"• Intervalle de mise à jour: {self.live_config['update_interval']}s")
            
            while True:
                try:
                    # 2. Mise à jour des données
                    updates = await self._update_live_data(timeframes)
                    if not updates:
                        continue
                    
                    # 3. Préparation des séquences
                    sequences = self._prepare_live_sequences(timeframes)
                    
                    # 4. Prédiction
                    prediction = await self._make_live_prediction(model, sequences)
                    
                    # 5. Logging et callback
                    self._log_live_prediction(prediction)
                    if callback:
                        await callback(prediction)
                    
                    # 6. Attente avant prochaine mise à jour
                    await asyncio.sleep(self.live_config['update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Erreur dans la boucle live: {str(e)}")
                    await asyncio.sleep(5)  # Attente courte avant retry
                    
        except Exception as e:
            self.logger.error(f"Erreur fatale dans live_predict: {str(e)}")
            raise

    async def _initialize_live_buffers(self, timeframes: List[str]) -> None:
        """Initialise les buffers avec données historiques"""
        try:
            for tf in timeframes:
                # Calculer la période de warmup
                warmup = self.live_config['warmup_periods'][tf]
                
                # Récupérer les données historiques
                df = await self.fetch_realtime_data(
                    symbol='XAUUSD',
                    timeframe=tf,
                    limit=warmup
                )
                
                if df is not None and not df.empty:
                    self.live_buffers[tf] = df
                    self.last_updates[tf] = df.index.max()
                    
                    self.logger.info(f"\nBuffer initialisé pour {tf}:")
                    self.logger.info(f"• Points: {len(df)}")
                    self.logger.info(f"• Période: {df.index.min()} à {df.index.max()}")
                else:
                    raise ValueError(f"Impossible d'initialiser le buffer pour {tf}")
                    
        except Exception as e:
            self.logger.error(f"Erreur dans _initialize_live_buffers: {str(e)}")
            raise

    async def _update_live_data(self, timeframes: List[str]) -> bool:
        """Met à jour les buffers avec nouvelles données"""
        try:
            updates_made = False
            
            for tf in timeframes:
                # Vérifier si une mise à jour est nécessaire
                if self._should_update(tf):
                    # Récupérer les nouvelles données
                    new_data = await self.fetch_realtime_data(
                        symbol='XAUUSD',
                        timeframe=tf,
                        limit=10  # Quelques barres pour éviter les gaps
                    )
                    
                    if new_data is not None and not new_data.empty:
                        # Mettre à jour le buffer
                        self.live_buffers[tf] = pd.concat([
                            self.live_buffers[tf],
                            new_data[~new_data.index.isin(self.live_buffers[tf].index)]
                        ]).sort_index()
                        
                        # Maintenir la taille du buffer
                        if len(self.live_buffers[tf]) > self.live_config['buffer_size']:
                            self.live_buffers[tf] = self.live_buffers[tf].iloc[
                                -self.live_config['buffer_size']:
                            ]
                        
                        self.last_updates[tf] = new_data.index.max()
                        updates_made = True
                        
                        self.logger.debug(f"Buffer {tf} mis à jour: {len(new_data)} points")
            
            return updates_made
            
        except Exception as e:
            self.logger.error(f"Erreur dans _update_live_data: {str(e)}")
            return False

    def _should_update(self, timeframe: str) -> bool:
        """Détermine si une mise à jour est nécessaire"""
        try:
            if timeframe not in self.last_updates:
                return True
                
            last_update = self.last_updates[timeframe]
            now = pd.Timestamp.now()
            
            # Intervalles de mise à jour par timeframe
            update_intervals = {
                '5m': pd.Timedelta(minutes=1),
                '15m': pd.Timedelta(minutes=3),
                '30m': pd.Timedelta(minutes=5),
                '1h': pd.Timedelta(minutes=10),
                '4h': pd.Timedelta(minutes=30)
            }
            
            return now - last_update > update_intervals[timeframe]
            
        except Exception as e:
            self.logger.error(f"Erreur dans _should_update: {str(e)}")
            return True

    def _prepare_live_sequences(self, timeframes: List[str]) -> Dict:
        """Prépare les séquences pour la prédiction"""
        try:
            sequences = {}
            
            for tf in timeframes:
                if tf not in self.live_buffers:
                    continue
                    
                # Extraire la dernière séquence
                data = self.live_buffers[tf].iloc[-self.live_config['sequence_length']:]
                
                if len(data) == self.live_config['sequence_length']:
                    # Normaliser les données
                    normalized = self._normalize_live_data(data)
                    sequences[f'input_{tf}'] = normalized
                
            return sequences
            
        except Exception as e:
            self.logger.error(f"Erreur dans _prepare_live_sequences: {str(e)}")
            return {}

    def _normalize_live_data(self, data: pd.DataFrame) -> np.ndarray:
        """Normalise les données pour le modèle"""
        try:
            # Utiliser la même normalisation que pour l'entraînement
            normalized = data.copy()
            
            # Normalisation des prix
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                normalized[col] = (normalized[col] - normalized[col].mean()) / normalized[col].std()
            
            # Normalisation du volume
            normalized['Volume'] = (normalized['Volume'] - normalized['Volume'].mean()) / normalized['Volume'].std()
            
            return normalized.values.reshape(1, -1, len(normalized.columns))
            
        except Exception as e:
            self.logger.error(f"Erreur dans _normalize_live_data: {str(e)}")
            return None

    async def _make_live_prediction(self, model, sequences: Dict) -> Dict:
        """Effectue la prédiction en temps réel"""
        try:
            if not sequences:
                return None
            
            # Prédiction
            prediction = model.predict(sequences)
            
            # Formatage du résultat
            result = {
                'timestamp': pd.Timestamp.now(),
                'prediction': np.argmax(prediction[0]) - 1,  # -1=Short, 0=Neutral, 1=Long
                'confidence': np.max(prediction[0]),
                'probabilities': {
                    'short': prediction[0][0],
                    'neutral': prediction[0][1],
                    'long': prediction[0][2]
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur dans _make_live_prediction: {str(e)}")
            return None

    def _log_live_prediction(self, prediction: Dict):
        """Log les détails de la prédiction"""
        if prediction:
            self.logger.info("\n📊 Nouvelle prédiction:")
            self.logger.info(f"• Timestamp: {prediction['timestamp']}")
            self.logger.info(f"• Signal: {prediction['prediction']}")
            self.logger.info(f"• Confiance: {prediction['confidence']:.1%}")
            self.logger.info("\nProbabilités:")
            for direction, prob in prediction['probabilities'].items():
                self.logger.info(f"• {direction}: {prob:.1%}")

class HistoricalDataCollector:
    def __init__(self):
        self.config = Config()
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Mapping des timeframes avec les noms de fichiers réels
        self.timeframe_mapping = {
            '5min': 'XAU_5m_data.csv',
            '15min': 'XAU_15m_data.csv',
            '30min': 'XAU_30m_data.csv',
            '1h': 'XAU_1h_data.csv',
            '4h': 'XAU_4h_data.csv',
            '1D': 'XAU_1d_data.csv',
            '1W': 'XAU_1w_data.csv',
            '1M': 'XAU_1Month_data.csv'
        }
        
        # Configuration de validation
        self.validation_config = {
            'min_data_points': {
                '5m': 100000,   # ~1 an de données
                '15m': 35000,   # ~1 an de données
                '30m': 17500,   # ~1 an de données
                '1h': 8760,     # ~1 an de données
                '4h': 2190,     # ~1 an de données
                '1D': 365,      # 1 an minimum
                '1W': 52,       # 1 an minimum
                '1M': 12        # 1 an minimum
            },
            'min_date': '2010-01-01',  # Date minimale acceptable
            'required_columns': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            'price_bounds': {
                'XAUUSD': {
                    'min': 1000.0,
                    'max': 3000.0
                }
            },
            'volume_bounds': {
                'min': 0,
                'max': 1e9  # Valeur maximale raisonnable
            }
        }

        # Configuration des retries
        self.retry_config = {
            'max_attempts': 3,
            'initial_wait': 2,  # secondes
            'max_wait': 60,     # secondes
            'backup_sources': {
                'local': 'data/backup',
                'url': 'https://backup-data.example.com/xauusd'  # URL exemple
            }
        }

    def list_available_files(self):
        """
        Liste tous les fichiers disponibles dans le dataset
        """
        dataset = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
        try:
            # Obtenir les métadonnées du dataset
            dataset_info = self.api.dataset_metadata(dataset)
            print("\nFichiers disponibles dans le dataset:")
            for file in dataset_info.files:
                print(f"- {file.name}")
            return dataset_info.files
        except Exception as e:
            print(f"[ERREUR] Erreur lors de la liste des fichiers: {str(e)}")
            return None
    
    def load_kaggle_data(self, timeframe):
        """Charge et valide les données depuis Kaggle"""
        try:
            # Téléchargement des données
            filepath = self._download_kaggle_data(timeframe)
            if not filepath:
                return None
            
            # Validation des données téléchargées
            validation_result = self._validate_downloaded_data(filepath, timeframe)
            
            if not validation_result['is_valid']:
                print("\n❌ Validation échouée:")
                for error in validation_result['errors']:
                    print(f"• {error}")
                return None
            
            print("\n✓ Validation réussie:")
            for stat in validation_result['stats']:
                print(f"• {stat}")
            
            return filepath
            
        except Exception as e:
            print(f"[ERREUR] Erreur lors du chargement des données: {str(e)}")
            return None

    def _validate_downloaded_data(self, filepath: str, timeframe: str) -> Dict:
        """Valide les données téléchargées"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'stats': []
            }
            
            # 1. Vérifier l'existence et la taille du fichier
            if not os.path.exists(filepath):
                validation_result['errors'].append(f"Fichier non trouvé: {filepath}")
                validation_result['is_valid'] = False
                return validation_result
            
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Taille minimale raisonnable
                validation_result['errors'].append(f"Fichier trop petit: {file_size} bytes")
                validation_result['is_valid'] = False
                return validation_result
            
            # 2. Lecture et validation du contenu
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except Exception as e:
                validation_result['errors'].append(f"Erreur de lecture CSV: {str(e)}")
                validation_result['is_valid'] = False
                return validation_result
            
            # 3. Vérification des colonnes requises
            missing_cols = set(self.validation_config['required_columns']) - set(df.columns)
            if missing_cols:
                validation_result['errors'].append(f"Colonnes manquantes: {missing_cols}")
                validation_result['is_valid'] = False
            
            # 4. Vérification du nombre de points
            min_points = self.validation_config['min_data_points'].get(timeframe, 1000)
            if len(df) < min_points:
                validation_result['errors'].append(
                    f"Nombre de points insuffisant: {len(df)} < {min_points}"
                )
                validation_result['is_valid'] = False
            
            # 5. Vérification de la période couverte
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                min_date = pd.to_datetime(self.validation_config['min_date'])
                
                if df['Date'].min() > min_date:
                    validation_result['errors'].append(
                        f"Période historique insuffisante: début {df['Date'].min()}"
                    )
                    validation_result['is_valid'] = False
                
                # Vérifier la continuité des données
                date_gaps = self._check_date_gaps(df['Date'], timeframe)
                if date_gaps['has_gaps']:
                    validation_result['errors'].append(
                        f"Gaps détectés: {date_gaps['gap_count']} gaps > {date_gaps['max_gap']} "
                        f"({date_gaps['gap_percentage']:.1%} des données)"
                    )
            except Exception as e:
                validation_result['errors'].append(f"Erreur validation dates: {str(e)}")
                validation_result['is_valid'] = False
            
            # 6. Validation des prix
            price_bounds = self.validation_config['price_bounds']['XAUUSD']
            for col in ['Open', 'High', 'Low', 'Close']:
                invalid_prices = (
                    (df[col] < price_bounds['min']) | 
                    (df[col] > price_bounds['max'])
                )
                if invalid_prices.any():
                    validation_result['errors'].append(
                        f"Prix {col} invalides: {invalid_prices.sum()} valeurs hors limites"
                    )
                    validation_result['is_valid'] = False
            
            # 7. Validation du volume
            invalid_volume = (
                (df['Volume'] < self.validation_config['volume_bounds']['min']) |
                (df['Volume'] > self.validation_config['volume_bounds']['max'])
            )
            if invalid_volume.any():
                validation_result['errors'].append(
                    f"Volumes invalides: {invalid_volume.sum()} valeurs hors limites"
                )
                validation_result['is_valid'] = False
            
            # 8. Statistiques si valide
            if validation_result['is_valid']:
                validation_result['stats'].extend([
                    f"Points de données: {len(df):,}",
                    f"Période: {df['Date'].min()} à {df['Date'].max()}",
                    f"Prix moyen: {df['Close'].mean():.2f}",
                    f"Volume moyen: {df['Volume'].mean():.0f}",
                    f"Taille fichier: {file_size/1024/1024:.1f} MB"
                ])
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"Erreur de validation: {str(e)}"],
                'stats': []
            }

    def _check_date_gaps(self, dates: pd.Series, timeframe: str) -> Dict:
        """Vérifie les gaps dans les données temporelles"""
        try:
            # Intervalles attendus par timeframe
            expected_intervals = {
                '5m': pd.Timedelta(minutes=5),
                '15m': pd.Timedelta(minutes=15),
                '30m': pd.Timedelta(minutes=30),
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                '1D': pd.Timedelta(days=1),
                '1W': pd.Timedelta(weeks=1),
                '1M': pd.Timedelta(days=30)
            }
            
            interval = expected_intervals.get(timeframe)
            if not interval:
                return {'has_gaps': False, 'gap_count': 0, 'max_gap': None}
            
            # Calcul des différences temporelles
            date_diff = dates.sort_values().diff()
            gaps = date_diff > (interval * 2)  # Gap = 2x l'intervalle normal
            
            if not gaps.any():
                return {'has_gaps': False, 'gap_count': 0, 'max_gap': None}
            
            return {
                'has_gaps': True,
                'gap_count': gaps.sum(),
                'max_gap': date_diff[gaps].max(),
                'gap_percentage': gaps.sum() / len(dates)
            }
            
        except Exception as e:
            print(f"Erreur dans _check_date_gaps: {str(e)}")
            return {'has_gaps': True, 'gap_count': -1, 'max_gap': None}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((ApiException, ConnectionError)),
        before_sleep=lambda retry_state: print(
            f"\n⚠️ Tentative {retry_state.attempt_number} échouée, "
            f"nouvelle tentative dans {retry_state.next_action.sleep} secondes..."
        )
    )
    def _download_kaggle_data(self, timeframe: str) -> Optional[str]:
        """Télécharge les données depuis Kaggle avec retry et fallback"""
        try:
            # 1. Vérifier d'abord le cache local
            cached_file = self._check_local_cache(timeframe)
            if cached_file:
                print(f"✓ Données trouvées en cache: {cached_file}")
                return cached_file
            
            # 2. Tentative de téléchargement Kaggle
            print(f"\nTéléchargement des données {timeframe} depuis Kaggle...")
            
            if timeframe not in self.timeframe_mapping:
                raise ValueError(
                    f"Timeframe '{timeframe}' non valide. "
                    f"Options: {list(self.timeframe_mapping.keys())}"
                )
            
            file_name = self.timeframe_mapping[timeframe]
            dataset = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
            output_dir = self.config.RAW_DATA_DIR
            
            # Créer le répertoire si nécessaire
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                self.api.dataset_download_files(
                    dataset,
                    path=output_dir,
                    unzip=True,
                    quiet=False
                )
            except ApiException as e:
                print(f"\n❌ Erreur API Kaggle: {str(e)}")
                if e.status == 403:  # Problème d'authentification
                    print("Vérification des credentials Kaggle...")
                    self._verify_kaggle_auth()
                raise
            
            # Vérifier le fichier téléchargé
            filepath = os.path.join(output_dir, file_name)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Fichier non trouvé après téléchargement: {file_name}")
            
            # Sauvegarder en cache
            self._update_local_cache(filepath, timeframe)
            
            print(f"✓ Téléchargement réussi: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"\n❌ Erreur de téléchargement: {str(e)}")
            
            # 3. Tentative de récupération depuis le backup
            backup_file = self._try_backup_sources(timeframe)
            if backup_file:
                print(f"✓ Données récupérées depuis backup: {backup_file}")
                return backup_file
            
            raise

    def _check_local_cache(self, timeframe: str) -> Optional[str]:
        """Vérifie si des données récentes existent en cache"""
        try:
            cache_dir = self.retry_config['backup_sources']['local']
            cache_file = os.path.join(cache_dir, self.timeframe_mapping[timeframe])
            
            if not os.path.exists(cache_file):
                return None
            
            # Vérifier l'âge du cache
            cache_age = time.time() - os.path.getmtime(cache_file)
            max_age = {
                '5m': 3600,    # 1 heure
                '15m': 3600,   # 1 heure
                '30m': 7200,   # 2 heures
                '1h': 14400,   # 4 heures
                '4h': 43200,   # 12 heures
                '1D': 86400,   # 24 heures
                '1W': 604800,  # 1 semaine
                '1M': 2592000  # 30 jours
            }.get(timeframe, 86400)
            
            if cache_age > max_age:
                return None
            
            return cache_file
            
        except Exception as e:
            print(f"Erreur lors de la vérification du cache: {str(e)}")
            return None

    def _update_local_cache(self, filepath: str, timeframe: str):
        """Met à jour le cache local"""
        try:
            cache_dir = self.retry_config['backup_sources']['local']
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, self.timeframe_mapping[timeframe])
            shutil.copy2(filepath, cache_file)
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour du cache: {str(e)}")

    def _try_backup_sources(self, timeframe: str) -> Optional[str]:
        """Tente de récupérer les données depuis les sources de backup"""
        try:
            # 1. Essayer le backup local
            backup_dir = self.retry_config['backup_sources']['local']
            backup_file = os.path.join(backup_dir, self.timeframe_mapping[timeframe])
            
            if os.path.exists(backup_file):
                return backup_file
            
            # 2. Essayer l'URL de backup
            backup_url = f"{self.retry_config['backup_sources']['url']}/{timeframe}"
            try:
                response = requests.get(backup_url, timeout=30)
                if response.status_code == 200:
                    # Sauvegarder la réponse
                    os.makedirs(backup_dir, exist_ok=True)
                    with open(backup_file, 'wb') as f:
                        f.write(response.content)
                    return backup_file
            except:
                pass
            
            return None
            
        except Exception as e:
            print(f"Erreur lors de la tentative de backup: {str(e)}")
            return None

    def _verify_kaggle_auth(self):
        """Vérifie et tente de corriger l'authentification Kaggle"""
        try:
            # Vérifier le fichier kaggle.json
            kaggle_dir = os.path.expanduser('~/.kaggle')
            kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
            
            if not os.path.exists(kaggle_file):
                # Tenter de créer depuis les variables d'environnement
                username = os.getenv('KAGGLE_USERNAME')
                key = os.getenv('KAGGLE_KEY')
                
                if username and key:
                    os.makedirs(kaggle_dir, exist_ok=True)
                    with open(kaggle_file, 'w') as f:
                        json.dump({
                            'username': username,
                            'key': key
                        }, f)
                    os.chmod(kaggle_file, 0o600)
                    
                    # Réinitialiser l'API
                    self.api = KaggleApi()
                    self.api.authenticate()
                else:
                    raise ValueError("Credentials Kaggle manquants")
            
        except Exception as e:
            print(f"Erreur lors de la vérification de l'authentification: {str(e)}")
            raise 