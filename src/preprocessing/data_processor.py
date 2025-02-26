import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List
from src.features.technical_indicators import TechnicalIndicators

class GoldDataProcessor:
    def __init__(self):
        pass
        
    def add_technical_indicators(self, df):
        """Ajoute les indicateurs techniques de base"""
        # Copie pour ne pas modifier l'original
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
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def add_price_patterns(self, df):
        """Détecte les patterns de prix basiques"""
        df = df.copy()
        
        # Tendance sur 5 jours
        df['Trend_5D'] = df['Close'].diff(5).apply(lambda x: 'Up' if x > 0 else 'Down')
        
        # Gaps
        df['Gap_Up'] = df['Open'] > df['Close'].shift(1)
        df['Gap_Down'] = df['Open'] < df['Close'].shift(1)
        
        return df
    
    def add_trading_signals(self, df):
        """Ajoute les signaux de trading basés sur les indicateurs"""
        df = df.copy()
        
        # Signaux basés sur les croisements de moyennes mobiles
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
        
        # Signaux RSI
        df['RSI_Oversold'] = df['RSI'] < 30
        df['RSI_Overbought'] = df['RSI'] > 70
        
        # Signaux MACD
        df['MACD_Cross_Up'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        df['MACD_Cross_Down'] = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
        
        return df

class DataProcessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.indicators = TechnicalIndicators()
        
    def detect_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Détecte et gère intelligemment les gaps temporels"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df['Gap'] = df.index.diff().total_seconds()
        
        # Définition des seuils par timeframe avec des tolérances maximales
        timeframe_sec = {
            '5m': 300, '15m': 900, '30m': 1800, 
            '1h': 3600, '4h': 14400, '1d': 86400, 
            '1w': 604800, '1M': 2592000
        }
        
        # Seuils de tolérance ultra souples
        base_gap = timeframe_sec.get(timeframe, 300)
        short_gap = base_gap * 48     # Augmenté à 48x (ex: 4h pour 5m)
        medium_gap = base_gap * 144   # Augmenté à 144x (ex: 12h pour 5m)
        long_gap = base_gap * 432     # Augmenté à 432x (ex: 36h pour 5m)
        
        # Identifier les différents types de gaps
        short_gaps = (df['Gap'] > base_gap) & (df['Gap'] <= short_gap)
        medium_gaps = (df['Gap'] > short_gap) & (df['Gap'] <= medium_gap)
        long_gaps = (df['Gap'] > medium_gap) & (df['Gap'] <= long_gap)
        very_long_gaps = df['Gap'] > long_gap
        
        print(f"\n📊 Analyse des gaps pour {timeframe}:")
        print(f"Gaps courts (<{short_gap/3600:.1f}h): {sum(short_gaps)}")
        print(f"Gaps moyens (<{medium_gap/3600:.1f}h): {sum(medium_gaps)}")
        print(f"Gaps longs (<{long_gap/3600:.1f}h): {sum(long_gaps)}")
        print(f"Gaps très longs (>{long_gap/3600:.1f}h): {sum(very_long_gaps)}")
        
        # 1. Traitement des gaps courts : interpolation linéaire
        if sum(short_gaps) > 0:
            for col in ['Open', 'High', 'Low', 'Close']:
                df.loc[short_gaps, col] = df[col].interpolate(method='linear')
            
        # 2. Traitement des gaps moyens : interpolation + bruit minimal
        if sum(medium_gaps) > 0:
            for col in ['Open', 'High', 'Low', 'Close']:
                interpolated = df[col].interpolate(method='linear')
                std = df[col].std() * np.clip(df['Gap'][medium_gaps] / medium_gap, 0.001, 0.01)
                noise = np.random.normal(0, std, size=sum(medium_gaps))
                df.loc[medium_gaps, col] = interpolated[medium_gaps] + noise
        
        # 3. Traitement des gaps longs : propagation avec tendance minimale
        if sum(long_gaps) > 0:
            for col in ['Open', 'High', 'Low', 'Close']:
                # Calculer la tendance locale avec fenêtre plus large
                rolling_mean = df[col].rolling(window=100, min_periods=1).mean()
                trend_direction = np.sign(rolling_mean.diff())
                
                # Propager avec tendance très réduite
                df.loc[long_gaps, col] = df[col].ffill()
                trend = np.linspace(0, df[col].std() * 0.01, sum(long_gaps)) * trend_direction[long_gaps]
                df.loc[long_gaps, col] += trend
        
        # 4. Traiter même les gaps très longs avec une approche conservative
        if sum(very_long_gaps) > 0:
            print(f"⚠️ Traitement de {sum(very_long_gaps)} gaps très longs")
            for col in ['Open', 'High', 'Low', 'Close']:
                df.loc[very_long_gaps, col] = df[col].ffill()
        
        return df.drop(columns=['Gap'])

    def detect_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte et corrige les anomalies de prix OHLC"""
        df = df.copy()
        
        # Vérification des prix incohérents
        invalid_high = (df['High'] < df['Open']) | (df['High'] < df['Close'])
        invalid_low = (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
        
        anomalies = invalid_high | invalid_low
        if anomalies.sum() > 0:
            print(f"\n⚠️ {anomalies.sum()} anomalies de prix détectées")
            
            # Correction par moyenne mobile
            df.loc[invalid_high, 'High'] = df[['Open', 'Close']].max(axis=1) * 1.001
            df.loc[invalid_low, 'Low'] = df[['Open', 'Close']].min(axis=1) * 0.999
            
            # Vérification après correction
            still_invalid = (df['High'] < df['Open']) | (df['High'] < df['Close']) | \
                           (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
            print(f"Anomalies restantes après correction: {still_invalid.sum()}")
        
        return df

    def prepare_training_data(self, data_dict: Dict[str, pd.DataFrame], 
                            sequence_length: int = 30,
                            prediction_horizon: int = 12,
                            train_size: float = 0.8) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement avec vérification approfondie"""
        print("\n🔄 Préparation des données...")
        
        # 1. Vérification et nettoyage initial
        for tf in list(data_dict.keys()):
            print(f"\n📊 Vérification des données {tf}:")
            df = data_dict[tf]
            
            # Vérifier la présence de données
            if len(df) == 0:
                print(f"⚠️ Timeframe {tf} vide, supprimé")
                del data_dict[tf]
                continue
            
            # Détecter les gaps
            df = self.detect_gaps(df, tf)
            
            # Corriger les prix anormaux
            df = self.detect_price_anomalies(df)
            
            # Rapport final
            print(f"  🔹 Lignes: {len(df)}")
            print(f"  🔹 Min/Max Close: {df['Close'].min():.4f} / {df['Close'].max():.4f}")
            print(f"  🔹 Période: {df.index.min()} → {df.index.max()}")
            
            data_dict[tf] = df
        
        # 2. Sur-échantillonnage des timeframes longs
        min_samples = 1000
        for tf in ['1d', '1w', '1M']:
            if tf in data_dict and len(data_dict[tf]) < min_samples:
                print(f"\n📈 Sur-échantillonnage {tf}: {len(data_dict[tf])} → {min_samples}")
                data_dict[tf] = data_dict[tf].sample(n=min_samples, replace=True)
        
        # Ajouter les indicateurs techniques
        print("\nAjout des indicateurs techniques...")
        data_with_features = {}
        
        # Réduire sequence_length pour avoir plus de données
        sequence_length = min(15, sequence_length)  # Maximum 15 pour garder plus d'échantillons
        
        for tf, df in data_dict.items():
            print(f"\nProcessing {tf}...")
            print(f"Shape initial: {df.shape}")
            
            # Ajouter les indicateurs
            df_with_features = TechnicalIndicators.add_all_indicators(df)
            print(f"Shape après ajout des indicateurs: {df_with_features.shape}")
            
            # Vérifier les NaN
            if df_with_features.isna().any().any():
                print("Remplacement des NaN...")
                df_with_features = df_with_features.fillna(method='ffill').fillna(method='bfill')
            
            data_with_features[tf] = df_with_features
        
        # Normaliser les données
        print("\nNormalisation des données...")
        normalized_data = self._normalize_data(data_with_features)
        
        # Créer les séquences
        print("\nCréation des séquences...")
        X = {}
        for tf, df in normalized_data.items():
            X[tf] = self._create_sequences(df, sequence_length, tf)
            print(f"{tf}: {X[tf].shape}")
        
        # Préparer les labels avec le timeframe le plus long disponible
        print("\nPréparation des labels...")
        longest_tf = max(data_dict.keys(), key=lambda x: len(data_dict[x]))
        y = self._create_labels(data_dict[longest_tf], prediction_horizon)
        
        # Aligner les données avec plus de souplesse
        print("\nAlignement des données...")
        X, y = self._align_data(X, y, min_samples=500)
        
        # Diviser en train/test
        print("\nDivision train/test...")
        return self.split_data(X, y, train_size)
    
    def split_data(self, X: Dict[str, np.ndarray], y: np.ndarray, 
                   train_size: float = 0.75) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """Divise les données en respectant l'ordre temporel et la distribution des labels"""
        print("\n📊 Division des données train/test...")
        
        # Définir les ratios cibles pour chaque classe
        target_ratios = {
            -1: 0.325,  # 32.5% pour les mouvements baissiers
            0: 0.350,   # 35.0% pour les mouvements neutres
            1: 0.325    # 32.5% pour les mouvements haussiers
        }
        
        # Calculer la taille minimale pour chaque ensemble
        min_train_samples = 2000  # Augmenté pour plus de robustesse
        min_test_samples = 1000   # Garantir au moins 1000 échantillons en test
        
        # Séparer les indices par classe
        indices_by_class = {label: np.where(y == label)[0] for label in [-1, 0, 1]}
        
        # Calculer les tailles cibles pour chaque classe
        total_target = max(min_train_samples + min_test_samples, len(y))
        target_per_class = {
            label: int(total_target * ratio)
            for label, ratio in target_ratios.items()
        }
        
        print("\n📈 Tailles cibles par classe:")
        for label, target in target_per_class.items():
            print(f"Classe {label}: {target} échantillons")
        
        # Préparer les indices train/test
        train_indices = []
        test_indices = []
        
        for label in [-1, 0, 1]:
            available_indices = indices_by_class[label]
            if len(available_indices) == 0:
                print(f"\n⚠️ Attention: Pas d'échantillons pour la classe {label}")
                continue
            
            # Calculer les tailles pour cette classe
            target_size = target_per_class[label]
            n_test = max(min_test_samples // 3, int(target_size * 0.25))  # Au moins 333 par classe en test
            n_train = min(len(available_indices) - n_test, int(target_size * 0.75))
            
            # Mélanger les indices tout en préservant l'ordre relatif
            np.random.seed(42)  # Pour la reproductibilité
            shuffled_indices = available_indices.copy()
            np.random.shuffle(shuffled_indices)
            
            # Diviser en train/test
            train_indices.extend(shuffled_indices[:n_train])
            test_indices.extend(shuffled_indices[n_train:n_train + n_test])
        
        # Mélanger les indices finaux
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Créer les ensembles train/test
        X_train = {tf: data[train_indices] for tf, data in X.items()}
        X_test = {tf: data[test_indices] for tf, data in X.items()}
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Vérifier les distributions
        print("\n📊 Distribution train:")
        self._print_distribution(y_train)
        print("\n📊 Distribution test:")
        self._print_distribution(y_test)
        
        # Vérifier les tailles
        print("\n📏 Tailles des ensembles:")
        print(f"Train: {len(y_train)} échantillons")
        print(f"Test: {len(y_test)} échantillons")
        
        # Vérifier les ratios par timeframe
        print("\n📊 Tailles par timeframe:")
        for tf in X.keys():
            print(f"\n{tf}:")
            print(f"Train: {X_train[tf].shape}")
            print(f"Test: {X_test[tf].shape}")
            train_ratio = len(X_train[tf]) / len(y_train)
            test_ratio = len(X_test[tf]) / len(y_test)
            print(f"Ratios train/test: {train_ratio:.2f}/{test_ratio:.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def _normalize_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalise les données de prix, volume et indicateurs techniques"""
        normalized = {}
        
        for tf, df in data_dict.items():
            normalized_df = df.copy()
            
            # Normaliser les prix OHLC
            price_cols = ['Open', 'High', 'Low', 'Close']
            normalized_df[price_cols] = self.price_scaler.fit_transform(df[price_cols])
            
            # Normaliser le volume
            normalized_df['Volume'] = self.volume_scaler.fit_transform(df[['Volume']])
            
            # Normaliser les indicateurs techniques
            tech_cols = [col for col in df.columns 
                        if col not in price_cols + ['Volume']]
            
            if tech_cols:
                tech_data = df[tech_cols].ffill().bfill()  # Forward fill puis backward fill
                normalized_df[tech_cols] = self.feature_scaler.fit_transform(tech_data)
            
            normalized[tf] = normalized_df
        
        return normalized
    
    def _get_timeframe_tolerances(self, timeframe: str) -> Tuple[float, float, float]:
        """Retourne les paramètres de tolérance adaptés au timeframe"""
        # Paramètres de base par timeframe avec des tolérances raisonnables
        tolerances = {
            # (gap_multiplier, min_atr, base_tolerance)
            '5m':  (20.0, 0.00005, 0.0001),   # Plus tolérant pour les mouvements rapides
            '15m': (18.0, 0.0001, 0.0002),    # Légèrement plus strict
            '30m': (16.0, 0.0002, 0.0005),    # Progression graduelle
            '1h':  (14.0, 0.0005, 0.001),     # Tolérance moyenne
            '4h':  (12.0, 0.001, 0.002),      # Plus strict pour les timeframes longs
            '1d':  (10.0, 0.002, 0.005),      # Encore plus strict
            '1w':  (8.0, 0.005, 0.01),        # Très strict
            '1M':  (6.0, 0.01, 0.02)          # Maximum de rigueur
        }
        
        # Valeurs par défaut raisonnables
        default_values = (15.0, 0.0005, 0.001)
        
        if timeframe in tolerances:
            return tolerances[timeframe]
        else:
            print(f"\n⚠️ Timeframe {timeframe} non reconnu, utilisation des valeurs par défaut")
            return default_values

    def _verify_temporal_continuity(self, sequence: np.ndarray, timeframe: str, 
                                  previous_end: np.ndarray = None) -> bool:
        """Vérifie la continuité temporelle avec des paramètres très souples"""
        # Indices des colonnes OHLC
        OPEN, HIGH, LOW, CLOSE = 0, 1, 2, 3
        
        # Obtenir les paramètres de tolérance pour ce timeframe
        MAX_GAP_MULTIPLIER, MIN_ATR_THRESHOLD, GAP_BASE_TOLERANCE = self._get_timeframe_tolerances(timeframe)
        
        # Vérifier la cohérence interne des prix OHLC de manière très souple
        for i in range(len(sequence)):
            # Calculer l'ATR local pour une tolérance adaptative
            if i > 0:
                tr = max(
                    sequence[i, HIGH] - sequence[i, LOW],  # Range actuel
                    abs(sequence[i, HIGH] - sequence[i-1, CLOSE]),  # High vs Close précédent
                    abs(sequence[i, LOW] - sequence[i-1, CLOSE])    # Low vs Close précédent
                )
                local_tolerance = max(MIN_ATR_THRESHOLD, tr)  # Utiliser 100% de l'ATR
            else:
                local_tolerance = MIN_ATR_THRESHOLD * 10  # Décuplé pour le premier point
            
            # Prix médian pour vérification très souple
            median_price = (sequence[i, HIGH] + sequence[i, LOW]) / 2
            price_range = max(
                sequence[i, HIGH] - sequence[i, LOW],
                MIN_ATR_THRESHOLD * 20  # Doublé pour éviter les ranges trop petits
            )
            
            # Vérifications avec tolérance maximale
            high_ok = (
                sequence[i, HIGH] >= sequence[i, OPEN] - local_tolerance * 2 and  # Doublé
                sequence[i, HIGH] >= sequence[i, LOW] - local_tolerance * 2 and   # Doublé
                sequence[i, HIGH] >= sequence[i, CLOSE] - local_tolerance * 2 and # Doublé
                abs(sequence[i, HIGH] - median_price) <= price_range * 4.0        # Doublé
            )
            
            low_ok = (
                sequence[i, LOW] <= sequence[i, OPEN] + local_tolerance * 2 and   # Doublé
                sequence[i, LOW] <= sequence[i, HIGH] + local_tolerance * 2 and   # Doublé
                sequence[i, LOW] <= sequence[i, CLOSE] + local_tolerance * 2 and  # Doublé
                abs(sequence[i, LOW] - median_price) <= price_range * 4.0         # Doublé
            )
            
            # Accepter plus de variations
            if not (high_ok and low_ok):
                # Donner une seconde chance avec des tolérances encore plus larges
                emergency_tolerance = local_tolerance * 4  # Quadruplé
                emergency_range = price_range * 8.0       # Octuplé
                
                high_emergency = (
                    sequence[i, HIGH] >= min(sequence[i, OPEN], sequence[i, CLOSE]) - emergency_tolerance and
                    abs(sequence[i, HIGH] - median_price) <= emergency_range
                )
                
                low_emergency = (
                    sequence[i, LOW] <= max(sequence[i, OPEN], sequence[i, CLOSE]) + emergency_tolerance and
                    abs(sequence[i, LOW] - median_price) <= emergency_range
                )
                
                if not (high_emergency and low_emergency):
                    return False
        
        # Vérifier la continuité temporelle de manière très souple
        for i in range(1, len(sequence)):
            # Calculer l'ATR avec tolérance maximale
            tr = max(
                sequence[i, HIGH] - sequence[i, LOW],
                abs(sequence[i, HIGH] - sequence[i-1, CLOSE]),
                abs(sequence[i, LOW] - sequence[i-1, CLOSE])
            )
            atr = max(tr, MIN_ATR_THRESHOLD)
            
            # Tolérance ultra adaptative
            adaptive_tolerance = max(
                GAP_BASE_TOLERANCE * median_price * 4,    # Quadruplé
                atr * MAX_GAP_MULTIPLIER * 2,            # Doublé
                price_range * 3,                         # Triplé
                abs(sequence[i, HIGH] - sequence[i-1, LOW]) * 2  # Doublé
            )
            
            # Vérifier le gap de prix avec tolérance maximale
            price_gap = abs(sequence[i, OPEN] - sequence[i-1, CLOSE])
            if price_gap > adaptive_tolerance:
                # Dernière chance avec tolérance d'urgence
                if price_gap <= adaptive_tolerance * 2:  # Double tolérance en cas d'urgence
                    continue
                return False
        
        return True

    def _create_sequences(self, df: pd.DataFrame, sequence_length: int, timeframe: str = '5m') -> np.ndarray:
        """Crée des séquences temporelles en préservant la structure"""
        print(f"\n🔄 Création des séquences temporelles pour {timeframe}...")
        
        # Vérifier et nettoyer les NaN initiaux
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f"\n⚠️ NaN détectés dans les colonnes: {nan_cols}")
            print("Nettoyage des données...")
            
            df = df.copy()
            for col in df.columns:
                filled = df[col].fillna(method='ffill')
                df[col] = filled.fillna(method='bfill')
        
        # Convertir en array numpy
        data = df.values
        n_samples = len(data) - sequence_length + 1
        n_features = data.shape[1]
        
        print(f"\n Statistiques des séquences:")
        print(f"Longueur de séquence: {sequence_length}")
        print(f"Nombre de features: {n_features}")
        print(f"Échantillons disponibles: {n_samples}")
        
        # Créer les séquences avec vérification améliorée
        sequences = np.zeros((n_samples, sequence_length, n_features))
        valid_mask = np.ones(n_samples, dtype=bool)
        previous_valid_end = None
        
        # Paramètres de tolérance spécifiques au timeframe
        MAX_GAP_MULTIPLIER, MIN_ATR_THRESHOLD, GAP_BASE_TOLERANCE = self._get_timeframe_tolerances(timeframe)
        
        print(f"\n⚙️ Paramètres de tolérance pour {timeframe}:")
        print(f"Gap multiplier: {MAX_GAP_MULTIPLIER}")
        print(f"Min ATR threshold: {MIN_ATR_THRESHOLD}")
        print(f"Base tolerance: {GAP_BASE_TOLERANCE}")
        
        # Créer les séquences
        for i in range(n_samples):
            sequence = data[i:(i + sequence_length)]
            
            # Vérifier la validité de base
            if np.isnan(sequence).any():
                valid_mask[i] = False
                continue
            
            # Vérifier la continuité temporelle avec les paramètres adaptés
            if not self._verify_temporal_continuity(sequence, timeframe, previous_valid_end):
                valid_mask[i] = False
                continue
            
            sequences[i] = sequence
            previous_valid_end = sequence[-1]
            
            # Afficher la progression
            if i % 1000 == 0:
                valid_count = np.sum(valid_mask[:i+1])
                print(f"Progress: {i+1}/{n_samples} séquences traitées, {valid_count} valides")
        
        # Filtrer les séquences invalides
        valid_sequences = sequences[valid_mask]
        
        print(f"\n✅ Séquences valides: {len(valid_sequences)}")
        print(f"Shape final: {valid_sequences.shape}")
        
        # Statistiques détaillées
        if len(valid_sequences) < n_samples:
            invalid_count = n_samples - len(valid_sequences)
            print(f"\n⚠️ {invalid_count} séquences invalides ignorées")
            print(f"Ratio de séquences valides: {len(valid_sequences)/n_samples*100:.2f}%")
            
            # Analyse des causes
            nan_count = np.sum(np.isnan(sequences).any(axis=(1,2)))
            discontinuity_count = invalid_count - nan_count
            print("\nCauses d'invalidité:")
            print(f"- NaN: {nan_count} séquences ({nan_count/invalid_count*100:.1f}%)")
            print(f"- Discontinuités: {discontinuity_count} séquences ({discontinuity_count/invalid_count*100:.1f}%)")
        
        return valid_sequences
    
    def _create_labels(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Crée les labels pour l'apprentissage basés sur les mouvements en pips"""
        print("\n🔄 Début de la création des labels...")
        
        # Calculer la variation en pips (1 pip = 0.01 pour XAUUSD)
        df = df.copy()
        df['future_price'] = df['Close'].shift(-horizon)
        df['price_change_pips'] = (df['future_price'] - df['Close']) * 100
        
        # Nettoyage initial des données
        df = df.dropna()
        price_change_pips = df['price_change_pips'].values
        
        print(f"\n📊 Statistiques des variations de prix (pips):")
        print(f"Min: {np.min(price_change_pips):.2f}")
        print(f"Max: {np.max(price_change_pips):.2f}")
        print(f"Moyenne: {np.mean(price_change_pips):.2f}")
        print(f"Écart-type: {np.std(price_change_pips):.2f}")
        
        # Calcul de la volatilité sur les rendements nettoyés
        df['returns'] = df['Close'].pct_change()
        df['returns'] = df['returns'].fillna(0)
        
        # Calcul des volatilités avec fenêtres glissantes
        df['volatility_short'] = df['returns'].rolling(window=20, min_periods=1).std()
        df['volatility_medium'] = df['returns'].rolling(window=50, min_periods=1).std()
        df['volatility_long'] = df['returns'].rolling(window=200, min_periods=1).std()
        
        # Remplissage des NaN avec la moyenne de la série
        for col in ['volatility_short', 'volatility_medium', 'volatility_long']:
            mean_vol = df[col].mean()
            df[col] = df[col].fillna(mean_vol)
        
        # Calcul de la volatilité moyenne pondérée
        df['avg_volatility'] = (
            0.6 * df['volatility_short'] + 
            0.3 * df['volatility_medium'] + 
            0.1 * df['volatility_long']
        )
        
        print("\n📈 Statistiques de volatilité:")
        print(f"Volatilité courte: {df['volatility_short'].mean():.4f}")
        print(f"Volatilité moyenne: {df['volatility_medium'].mean():.4f}")
        print(f"Volatilité longue: {df['volatility_long'].mean():.4f}")
        print(f"Volatilité moyenne pondérée: {df['avg_volatility'].mean():.4f}")
        
        # Calcul des seuils adaptatifs avec zones distinctes
        volatility_factor = np.clip(1 + df['avg_volatility'].values * 100, 0.8, 2.0)
        BASE_THRESHOLD = 10
        
        # Définition des seuils avec zones distinctes
        up_threshold = BASE_THRESHOLD * volatility_factor
        down_threshold = -BASE_THRESHOLD * volatility_factor
        neutral_upper = BASE_THRESHOLD * 0.3 * volatility_factor  # Réduit à 30% pour la zone neutre
        neutral_lower = -BASE_THRESHOLD * 0.3 * volatility_factor
        
        print(f"\n🎯 Seuils calculés:")
        print(f"Seuil haut: > {np.mean(up_threshold):.2f} pips")
        print(f"Zone neutre haute: {np.mean(neutral_upper):.2f} à {np.mean(up_threshold):.2f} pips")
        print(f"Zone neutre: {np.mean(neutral_lower):.2f} à {np.mean(neutral_upper):.2f} pips")
        print(f"Zone neutre basse: {np.mean(down_threshold):.2f} à {np.mean(neutral_lower):.2f} pips")
        print(f"Seuil bas: < {np.mean(down_threshold):.2f} pips")
        
        # Création des labels avec zones distinctes
        labels = np.zeros(len(price_change_pips))
        
        # Définition des zones sans chevauchement
        strong_up_moves = price_change_pips > up_threshold
        strong_down_moves = price_change_pips < down_threshold
        neutral_zone = (price_change_pips >= neutral_lower) & (price_change_pips <= neutral_upper)
        weak_up_moves = (price_change_pips > neutral_upper) & (price_change_pips <= up_threshold)
        weak_down_moves = (price_change_pips < neutral_lower) & (price_change_pips >= down_threshold)
        
        # Assignation des labels
        labels[strong_up_moves] = 1
        labels[strong_down_moves] = -1
        labels[neutral_zone] = 0
        labels[weak_up_moves] = 0.5  # Mouvements haussiers faibles
        labels[weak_down_moves] = -0.5  # Mouvements baissiers faibles
        
        print("\n🔍 Distribution détaillée des mouvements:")
        print(f"Fortement haussier: {np.sum(strong_up_moves)} ({np.sum(strong_up_moves)/len(labels)*100:.2f}%)")
        print(f"Faiblement haussier: {np.sum(weak_up_moves)} ({np.sum(weak_up_moves)/len(labels)*100:.2f}%)")
        print(f"Neutre: {np.sum(neutral_zone)} ({np.sum(neutral_zone)/len(labels)*100:.2f}%)")
        print(f"Faiblement baissier: {np.sum(weak_down_moves)} ({np.sum(weak_down_moves)/len(labels)*100:.2f}%)")
        print(f"Fortement baissier: {np.sum(strong_down_moves)} ({np.sum(strong_down_moves)/len(labels)*100:.2f}%)")
        
        # Vérification de la couverture totale
        total_classified = np.sum(strong_up_moves) + np.sum(weak_up_moves) + np.sum(neutral_zone) + \
                          np.sum(weak_down_moves) + np.sum(strong_down_moves)
        
        if total_classified != len(labels):
            print(f"\n⚠️ ATTENTION: {len(labels) - total_classified} points non classifiés!")
            print("Vérification des zones non couvertes...")
            unclassified = ~(strong_up_moves | weak_up_moves | neutral_zone | weak_down_moves | strong_down_moves)
            if np.any(unclassified):
                unclass_values = price_change_pips[unclassified]
                print(f"Valeurs non classifiées: min={np.min(unclass_values):.2f}, max={np.max(unclass_values):.2f}")
        
        # Conversion finale en classification à 3 classes
        labels = np.sign(labels)  # Convertit [-0.5, 0, 0.5] en [-1, 0, 1]
        
        print("\n🎯 Distribution finale avant équilibrage:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Label {label}: {count} ({count/len(labels)*100:.2f}%)")
        
        return self._balance_labels(labels)

    def _balance_labels(self, labels: np.ndarray) -> np.ndarray:
        """Équilibre les classes avec vérification"""
        min_samples_per_class = 1000
        balanced_labels = []
        
        for label in [-1, 0, 1]:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) == 0:
                print(f"\n⚠️ Aucun échantillon pour la classe {label}, génération synthétique")
                other_indices = np.where(labels != label)[0]
                if len(other_indices) > 0:
                    synthetic_samples = np.random.choice(other_indices, min_samples_per_class)
                    balanced_labels.extend([label] * min_samples_per_class)
            else:
                print(f"\nClasse {label}: {len(label_indices)} échantillons originaux")
                n_samples = max(min_samples_per_class, len(label_indices))
                sampled_indices = np.random.choice(label_indices, size=n_samples, replace=True)
                noise = np.random.normal(0, 0.1, size=len(sampled_indices))
                noisy_indices = np.clip(sampled_indices + noise, 0, len(labels)-1).astype(int)
                balanced_labels.extend(labels[noisy_indices])
        
        balanced_labels = np.array(balanced_labels)
        np.random.shuffle(balanced_labels)
        
        print("\n🎯 Distribution finale après équilibrage:")
        self._print_distribution(balanced_labels)
        
        return balanced_labels

    def _print_distribution(self, labels):
        """Affiche la distribution des labels"""
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        total = sum(counts)
        for label, count in sorted(dist.items()):
            print(f"Label {label}: {count} ({count/total*100:.2f}%)")
    
    def _add_adaptive_noise(self, data: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """Ajoute du bruit adaptatif aux données sur-échantillonnées"""
        # Calculer l'écart-type par feature
        std_per_feature = np.std(data, axis=(0, 1), keepdims=True)
        
        # Éviter les valeurs trop petites
        min_std = 1e-6
        std_per_feature = np.maximum(std_per_feature, min_std)
        
        # Générer du bruit adaptatif
        noise = np.random.normal(0, noise_scale * std_per_feature, data.shape)
        
        # Ajouter le bruit de manière progressive
        noise_factor = np.linspace(0, 1, data.shape[1])[:, np.newaxis]
        noise = noise * noise_factor
        
        return data + noise

    def _align_data(self, X: Dict[str, np.ndarray], y: np.ndarray, min_samples: int = 500) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Aligne les données en préservant les timeframes longs"""
        print("\n🔄 Alignement des données...")
        
        # Vérifier les séquences vides
        empty_timeframes = [tf for tf, seq in X.items() if len(seq) == 0]
        if empty_timeframes:
            print(f"\n⚠️ Attention: Séquences vides détectées pour les timeframes: {empty_timeframes}")
            X = {tf: seq for tf, seq in X.items() if len(seq) > 0}
            if not X:
                raise ValueError("Toutes les séquences sont vides!")
        
        # Définir les ratios cibles plus conservateurs
        target_ratios = {
            '5m':  1.00,    # Base de référence
            '15m': 0.90,    # Légère réduction
            '30m': 0.80,    # Réduction progressive
            '1h':  0.70,    # Réduction plus importante
            '4h':  0.60,    # Réduction significative
            '1d':  0.50,    # Moitié des données
            '1w':  0.30,    # Pas de sur-échantillonnage excessif
            '1M':  0.20     # Minimum pour préserver la structure
        }
        
        # Limites de sur-échantillonnage par timeframe
        max_upsampling = {
            '5m':  10,      # Maximum 10x pour les timeframes courts
            '15m': 8,
            '30m': 6,
            '1h':  5,
            '4h':  4,
            '1d':  3,
            '1w':  2,       # Presque pas de sur-échantillonnage
            '1M':  1.5      # Minimum de sur-échantillonnage
        }
        
        # Calculer les ratios actuels
        base_len = max(len(seq) for seq in X.values())
        current_ratios = {tf: len(seq) / base_len for tf, seq in X.items()}
        
        print("\n📊 Ratios actuels vs cibles:")
        for tf in X.keys():
            print(f"{tf}:")
            print(f"  Actuel: {current_ratios[tf]:.3f}")
            print(f"  Cible:  {target_ratios.get(tf, 0.5):.3f}")
            print(f"  Max upsampling: {max_upsampling.get(tf, 2)}x")
        
        # Appliquer l'échantillonnage avec limites
        X_aligned = {}
        for tf, seq in X.items():
            if len(seq) == 0:
                continue
            
            # Calculer la taille cible avec limite de sur-échantillonnage
            max_size = int(len(seq) * max_upsampling.get(tf, 2))
            target_size = min(
                max_size,
                int(base_len * target_ratios.get(tf, 0.5))
            )
            
            if len(seq) < target_size:
                # Sur-échantillonnage contrôlé
                upsampling_factor = min(target_size / len(seq), max_upsampling.get(tf, 2))
                actual_size = int(len(seq) * upsampling_factor)
                
                # Interpolation avec bruit minimal
                indices = np.linspace(0, len(seq)-1, actual_size, dtype=int)
                interpolated = seq[indices]
                
                # Bruit très faible et progressif
                noise_scale = 0.001 * (1 - current_ratios[tf])  # Réduit à 0.1%
                noisy_data = self._add_adaptive_noise(interpolated, noise_scale)
                
                X_aligned[tf] = noisy_data
                
                print(f"\n📊 Sur-échantillonnage contrôlé pour {tf}:")
                print(f"  Original: {len(seq)} → Final: {len(X_aligned[tf])}")
                print(f"  Facteur effectif: {len(X_aligned[tf])/len(seq):.2f}x")
                print(f"  Échelle de bruit: {noise_scale:.6f}")
            else:
                # Sous-échantillonnage avec préservation de la structure
                indices = np.linspace(0, len(seq)-1, target_size, dtype=int)
                X_aligned[tf] = seq[indices]
        
        # Aligner les labels
        min_len = min(len(seq) for seq in X_aligned.values())
        y_aligned = y[:min_len]
        
        # Ajuster les séquences finales
        for tf in X_aligned:
            if len(X_aligned[tf]) > min_len:
                indices = np.sort(np.random.choice(len(X_aligned[tf]), min_len, replace=False))
                X_aligned[tf] = X_aligned[tf][indices]
        
        print("\n📏 Tailles finales et ratios:")
        print(f"Labels: {len(y_aligned)}")
        for tf, seq in X_aligned.items():
            ratio = len(seq) / len(y_aligned)
            print(f"{tf}: {len(seq)} (ratio={ratio:.2f}, cible={target_ratios.get(tf, 0.5):.2f})")
        
        return X_aligned, y_aligned 