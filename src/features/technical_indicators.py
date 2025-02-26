import pandas as pd
import numpy as np
from typing import List

class TechnicalIndicators:
    """Classe pour calculer les indicateurs techniques"""
    
    @staticmethod
    def SMA(df: pd.DataFrame, window: int) -> pd.Series:
        """Calcule la moyenne mobile simple"""
        return df['Close'].rolling(window=window).mean()
    
    @staticmethod
    def EMA(df: pd.DataFrame, window: int) -> pd.Series:
        """Calcule la moyenne mobile exponentielle"""
        return df['Close'].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def RSI(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def MACD(df: pd.DataFrame) -> pd.DataFrame:
        """Calcule le MACD"""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        })
    
    @staticmethod
    def Stochastic(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calcule l'oscillateur stochastique"""
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=3).mean()
        return pd.DataFrame({
            '%K': k,
            '%D': d
        })
    
    @staticmethod
    def ATR(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule l'Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def BollingerBands(df: pd.DataFrame, window: int = 20, std: int = 2) -> pd.DataFrame:
        """Calcule les bandes de Bollinger"""
        middle = df['Close'].rolling(window=window).mean()
        std_dev = df['Close'].rolling(window=window).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return pd.DataFrame({
            'Middle': middle,
            'Upper': upper,
            'Lower': lower
        })
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les moyennes mobiles"""
        df = df.copy()
        # SMA
        for window in [10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # EMA
        for window in [12, 26]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de momentum"""
        df = df.copy()
        
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
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de volatilité"""
        df = df.copy()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + df['Close'].rolling(window=20).std() * 2
        df['BB_Lower'] = df['BB_Middle'] - df['Close'].rolling(window=20).std() * 2
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de volume"""
        # Volume moyen
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques avec gestion des NaN"""
        df = df.copy()
        
        # Moyennes mobiles avec différentes périodes
        df['SMA_10'] = TechnicalIndicators.SMA(df, 10)
        df['SMA_20'] = TechnicalIndicators.SMA(df, 20)
        df['SMA_50'] = TechnicalIndicators.SMA(df, 50)
        df['SMA_200'] = TechnicalIndicators.SMA(df, 200)
        
        # Moyennes mobiles exponentielles
        df['EMA_12'] = TechnicalIndicators.EMA(df, 12)
        df['EMA_26'] = TechnicalIndicators.EMA(df, 26)
        
        # RSI
        df['RSI'] = TechnicalIndicators.RSI(df)
        
        # MACD
        macd_data = TechnicalIndicators.MACD(df)
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        df['MACD_Hist'] = macd_data['Histogram']
        
        # Stochastique
        stoch_data = TechnicalIndicators.Stochastic(df)
        df['%K'] = stoch_data['%K']
        df['%D'] = stoch_data['%D']
        
        # ATR
        df['ATR'] = TechnicalIndicators.ATR(df)
        
        # Bandes de Bollinger
        bb_data = TechnicalIndicators.BollingerBands(df)
        df['BB_Middle'] = bb_data['Middle']
        df['BB_Upper'] = bb_data['Upper']
        df['BB_Lower'] = bb_data['Lower']
        
        # Gérer les NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df 