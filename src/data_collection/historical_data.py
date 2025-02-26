import os
import pandas as pd
from config.config import Config
from kaggle.api.kaggle_api_extended import KaggleApi

class HistoricalDataLoader:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.timeframes = {
            '5m': 'XAU_5m_data.csv',
            '15m': 'XAU_15m_data.csv',
            '30m': 'XAU_30m_data.csv',
            '1h': 'XAU_1h_data.csv',
            '4h': 'XAU_4h_data.csv',
            '1d': 'XAU_1d_data.csv',
            '1w': 'XAU_1w_data.csv',
            '1M': 'XAU_1Month_data.csv'
        }
        
    def load_multi_timeframe_data(self):
        """Charge et aligne les données de différents timeframes"""
        data = {}
        missing_files = []
        
        # Vérifier et télécharger les fichiers manquants
        print("\nVérification des fichiers:")
        for tf, filename in self.timeframes.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Fichier manquant: {filename}")
                try:
                    print(f"Tentative de téléchargement depuis Kaggle...")
                    self._download_data(tf)
                except Exception as e:
                    print(f"Erreur lors du téléchargement: {str(e)}")
                    missing_files.append(tf)
            else:
                print(f"Fichier trouvé: {filename}")
        
        # Charger les données
        for tf, filename in self.timeframes.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    print(f"\nChargement des données {tf}...")
                    # Format standard pour tous les fichiers
                    df = pd.read_csv(filepath, sep=None, engine='python')
                    
                    # Conversion des types
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
                    df.set_index('Date', inplace=True)
                    
                    # Convertir les colonnes numériques
                    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    data[tf] = df
                    print(f"Chargé avec succès: {tf} - Shape: {df.shape}")
                    
                except Exception as e:
                    print(f"Erreur lors du chargement de {filename}: {str(e)}")
                    missing_files.append(tf)
        
        if missing_files:
            print(f"\nFichiers non disponibles: {', '.join(missing_files)}")
            print("Note: Continuation avec les fichiers disponibles")
        
        if not data:
            print("Aucun fichier n'a pu être chargé")
            return None
        
        # Aligner les données sur les dates communes
        start_date = max(df.index.min() for df in data.values())
        end_date = min(df.index.max() for df in data.values())
        
        print(f"\nPériode commune : {start_date} à {end_date}")
        
        # Filtrer chaque dataframe pour la période commune
        aligned_data = {}
        for tf, df in data.items():
            aligned_df = df.loc[start_date:end_date]
            
            # Vérifier qu'il n'y a pas de données manquantes
            if aligned_df.isnull().any().any():
                print(f"Attention: données manquantes dans {tf}")
                aligned_df = aligned_df.fillna(method='ffill')
            
            aligned_data[tf] = aligned_df
            print(f"Shape final {tf}: {aligned_df.shape}")
        
        return aligned_data
    
    def _download_data(self, timeframe):
        """Télécharge les données depuis Kaggle"""
        try:
            api = KaggleApi()
            api.authenticate()
            
            # Créer le répertoire si nécessaire
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Télécharger le dataset
            dataset = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
            print(f"Téléchargement depuis {dataset}...")
            
            # Sauvegarder dans un dossier temporaire d'abord
            temp_dir = os.path.join(self.data_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            api.dataset_download_files(
                dataset,
                path=temp_dir,
                unzip=True,
                quiet=False
            )
            
            # Déplacer les fichiers vers le dossier final
            source_file = os.path.join(temp_dir, self.timeframes[timeframe])
            target_file = os.path.join(self.data_dir, self.timeframes[timeframe])
            
            if os.path.exists(source_file):
                try:
                    if not os.path.exists(target_file):
                        import shutil
                        shutil.copy2(source_file, target_file)
                    print(f"Données {timeframe} téléchargées avec succès")
                except PermissionError:
                    print(f"Note: Le fichier {self.timeframes[timeframe]} existe déjà et est peut-être utilisé")
            
            # Nettoyer le dossier temporaire
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Erreur lors du téléchargement: {str(e)}")
            raise

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
            print(f"Erreur lors de la liste des fichiers: {str(e)}")
            return None
    
    def load_kaggle_data(self, timeframe):
        """
        Charge les données historiques depuis Kaggle
        timeframe: '5min', '15min', '30min', '1h', '4h', '1D', '1W', '1M'
        """
        try:
            # Obtenir le nom correct du fichier
            if timeframe not in self.timeframe_mapping:
                raise ValueError(f"Timeframe '{timeframe}' non valide. Options disponibles: {list(self.timeframe_mapping.keys())}")
            
            file_name = self.timeframe_mapping[timeframe]
            dataset = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
            output_dir = self.config.RAW_DATA_DIR
            
            print(f"Téléchargement du fichier {file_name} depuis Kaggle...")
            
            # Télécharger le dataset complet
            self.api.dataset_download_files(
                dataset,
                path=output_dir,
                unzip=True,
                quiet=False
            )
            
            # Le fichier garde son nom et format original
            filepath = os.path.join(output_dir, file_name)
            if os.path.exists(filepath):
                print(f"Fichier trouvé : {filepath}")
                return filepath
            else:
                print(f"Fichiers disponibles dans {output_dir}:")
                for f in os.listdir(output_dir):
                    print(f"- {f}")
                raise FileNotFoundError(f"Le fichier {file_name} n'a pas été trouvé dans le dataset")
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            print("Type d'erreur:", type(e).__name__)
            print("Détails:", str(e))
            return None 