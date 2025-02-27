import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

class TechnicalIndicators:
    """Classe optimisée pour le calcul des indicateurs techniques essentiels"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration des indicateurs
        self.config = {
            'ema': {'periods': [9, 21, 50]},
            'rsi': {'period': 14},
            'bollinger': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        }

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques essentiels"""
        try:
            df = df.copy()
            
            # EMAs
            for period in self.config['ema']['periods']:
                df[f'EMA_{period}'] = self._calculate_ema(df, period)
            
            # RSI
            df['RSI'] = self._calculate_rsi(df)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df)
            df['BB_Upper'] = bb_data['Upper']
            df['BB_Middle'] = bb_data['Middle']
            df['BB_Lower'] = bb_data['Lower']
            
            # ATR
            df['ATR'] = self._calculate_atr(df)
            
            # MACD
            macd_data = self._calculate_macd(df)
            df['MACD'] = macd_data['MACD']
            df['MACD_Signal'] = macd_data['Signal']
            df['MACD_Hist'] = macd_data['Histogram']
            
            # OBV
            df['OBV'] = self._calculate_obv(df)
            
            # Nettoyage final
            return self._clean_data(df)
            
        except Exception as e:
            self.logger.error(f"Erreur dans calculate_all: {str(e)}")
            return df

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcule l'EMA"""
        return df['Close'].ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calcule le RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi']['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi']['period']).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcule les bandes de Bollinger"""
        period = self.config['bollinger']['period']
        std_dev = self.config['bollinger']['std_dev']
        
        middle = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        return {
            'Upper': middle + (std * std_dev),
            'Middle': middle,
            'Lower': middle - (std * std_dev)
        }

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calcule l'ATR"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=self.config['atr']['period']).mean()

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcule le MACD"""
        fast = self.config['macd']['fast']
        slow = self.config['macd']['slow']
        signal = self.config['macd']['signal']
        
        fast_ema = df['Close'].ewm(span=fast, adjust=False).mean()
        slow_ema = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return {
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': macd - signal_line
        }

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calcule l'OBV"""
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données en gérant les NaN"""
        # Remplacer les infinis par NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill puis backward fill pour les NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df 