import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import sys
import os

# Ajout du chemin racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import explicite depuis le module
from src.preprocessing.data_processor import DataProcessor

class MLPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data_processor = DataProcessor()
        
    def prepare_features(self, df):
        """Prépare les features pour le ML"""
        df = df.copy()
        
        # Calculer d'abord les moyennes mobiles
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Features de volatilité améliorées
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(window=window).std()
            df[f'Range_{window}'] = df['High'].rolling(window=window).max() - df['Low'].rolling(window=window).min()
            df[f'Price_Momentum_{window}'] = df['Close'].pct_change(window)
        
        # Features de base
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Distance_from_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Distance_from_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Momentum
        df['ROC'] = df['Close'].pct_change(10)  # Rate of Change
        df['MOM'] = df['Close'] - df['Close'].shift(10)  # Momentum
        
        # Volatilité
        df['ATR'] = self.calculate_atr(df)  # Average True Range
        df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df)
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Ajouter des features de volatilité avancées
        for window in [5, 10, 20, 30]:
            # Volatilité classique
            df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(window=window).std()
            
            # True Range normalisé
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'TR_{window}'] = tr.rolling(window=window).mean()
            
            # Momentum avec volatilité
            momentum = df['Close'].pct_change(window)
            volatility = df['Close'].pct_change().rolling(window=window).std()
            df[f'Vol_Momentum_{window}'] = momentum * (1 + volatility)
        
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD',
            'Price_Change', 'Volatility',
            'Distance_from_SMA20', 'Distance_from_SMA50',
            'ROC', 'MOM', 'ATR',
            'BB_Upper', 'BB_Lower',
            'Volume_Ratio',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'Range_5', 'Range_10', 'Range_20',
            'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20',
            'TR_5', 'TR_10', 'TR_20', 'TR_30',
            'Vol_Momentum_5', 'Vol_Momentum_10', 'Vol_Momentum_20', 'Vol_Momentum_30'
        ]
        
        # Ajouter les indicateurs manquants
        df['RSI'] = self.calculate_rsi(df)
        df['MACD'] = self.calculate_macd(df)
        
        df = df.dropna()
        X = self.scaler.fit_transform(df[features])
        return pd.DataFrame(X, columns=features, index=df.index)
    
    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        sma = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, lower_band
    
    def prepare_target(self, df, horizon=5):
        """Prépare la variable cible"""
        df = df.copy()
        
        # Calculer la variation de prix future
        df['Target'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
        
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna(subset=['Target'])
        
        return df['Target']
    
    def create_sequences(self, X, y, sequence_length=10):
        """Crée des séquences pour les modèles de séries temporelles"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X.iloc[i:i+sequence_length].values)
            y_seq.append(y.iloc[i+sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_multi_timeframe_features(self, data_dict):
        """Prépare les features pour chaque timeframe"""
        processed_data = {}
        
        for timeframe, df in data_dict.items():
            print(f"Traitement du timeframe {timeframe}...")
            df = df.copy()
            
            # Calculer les features
            df = self.add_technical_indicators(df)
            
            # Supprimer les colonnes de base
            features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            
            # Remplacer les infinis par NaN
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Remplir les NaN avec ffill puis bfill
            features = features.ffill().bfill()
            
            # Normaliser les features
            features_array = self.scaler.fit_transform(features)
            features = pd.DataFrame(
                features_array,
                columns=features.columns,
                index=features.index
            )
            
            processed_data[timeframe] = features
            print(f"Features créées pour {timeframe}: {features.shape[1]} colonnes")
        
        return processed_data
    
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
            print(f"Création des séquences pour {timeframe}...")
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
                print(f"Séquences {timeframe}: {len(X_seq)} échantillons")
            else:
                print(f"Attention: Aucune séquence créée pour {timeframe}")
        
        return sequences
    
    def add_technical_indicators(self, df):
        """Ajoute les indicateurs techniques de base"""
        df = df.copy()
        
        # Moyennes mobiles
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # Stochastique
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Momentum
        df['Momentum'] = df['Close'].pct_change(periods=10)
        
        # Force relative
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Volumes
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
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
        print("\nDistribution finale des données:")
        print("\nTrain set:")
        self._print_distribution(y_train)
        print("\nTest set:")
        self._print_distribution(y_test)
        
        return X_train, y_train, X_test, y_test

    def _print_distribution(self, y):
        """Affiche la distribution détaillée des labels"""
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        total = sum(counts)
        
        print("Distribution des classes:")
        for label, count in sorted(dist.items()):
            percentage = (count/total) * 100
            print(f"Classe {label:2}: {count:4d} échantillons ({percentage:5.2f}%)")
        
        # Calculer des statistiques supplémentaires
        if len(unique) > 1:
            max_to_min_ratio = max(counts) / min(counts)
            print(f"\nRatio max/min: {max_to_min_ratio:.2f}")
            print(f"Écart-type des proportions: {np.std(counts/total):.3f}")

    def check_data_balance(self, y_train, y_test):
        """Vérifie l'équilibre des données train/test"""
        print("\nVérification de l'équilibre des données:")
        
        # Distribution train
        print("\nEnsemble d'entraînement:")
        self._print_distribution(y_train)
        
        # Distribution test
        print("\nEnsemble de test:")
        self._print_distribution(y_test)
        
        # Vérifier les ratios
        train_dist = np.bincount(y_train.astype(int) + 1) / len(y_train)
        test_dist = np.bincount(y_test.astype(int) + 1) / len(y_test)
        
        print("\nDifférence de distribution train/test:")
        for i, (train_pct, test_pct) in enumerate(zip(train_dist, test_dist)):
            diff = abs(train_pct - test_pct) * 100
            print(f"Classe {i-1}: {diff:.2f}% de différence")
        
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