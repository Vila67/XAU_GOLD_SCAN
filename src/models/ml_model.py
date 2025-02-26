import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time
import backtrader as bt
from datetime import datetime
import os
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from typing import Dict, List

class MLModel:
    """Modèle d'apprentissage profond pour le trading"""
    
    def __init__(self, input_shape=None, n_classes=3, model_type='auto', learning_rate=0.001,
                 lstm_units=64, dense_units=32, dropout_rate=0.2, batch_size=32,
                 memory_efficient=False):
        """
        Initialise le modèle
        
        Args:
            input_shape (tuple): Forme des données d'entrée
            n_classes (int): Nombre de classes
            model_type (str): Type de modèle ('auto', 'transformer' ou 'lstm')
            learning_rate (float): Taux d'apprentissage
            lstm_units (int): Nombre d'unités LSTM
            dense_units (int): Nombre d'unités denses
            dropout_rate (float): Taux de dropout
            batch_size (int): Taille du batch
            memory_efficient (bool): Si le modèle doit être optimisé pour la mémoire
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.memory_efficient = memory_efficient
        
        # Scalers pour les features et les prix
        self.feature_scaler = StandardScaler()
        self.price_scalers = {
            'XAUUSD': {
                'scaler': StandardScaler(),
                'base_price': 1800.0,  # Prix de référence pour l'or
                'min_price': 1000.0,   # Prix minimum attendu
                'max_price': 3000.0,   # Prix maximum attendu
                'pip_value': 0.01      # Valeur d'un pip
            }
        }
        self.current_instrument = 'XAUUSD'  # Instrument par défaut
        
        # Standardisation des noms de timeframes
        self.timeframe_keys = {
            '5m': 'input_5m',
            '15m': 'input_15m',
            '30m': 'input_30m',
            '1h': 'input_1h',
            '4h': 'input_4h',
            '1d': 'input_1d',
            '1w': 'input_1w',
            '1M': 'input_1M'
        }
        
        # Mise à jour de la configuration avec les clés standardisées
        self.timeframe_config = {
            self.timeframe_keys['5m']: {
                'lstm_units': [128, 64],
                'dropout': [0.3, 0.3],
                'weight': 0.25,
                'samples_per_day': 288,
                'volatility_factor': 1.0
            },
            self.timeframe_keys['15m']: {
                'lstm_units': [128, 64],
                'dropout': [0.3, 0.3],
                'weight': 0.20,
                'samples_per_day': 96,
                'volatility_factor': 1.2
            },
            self.timeframe_keys['30m']: {
                'lstm_units': [128, 64],
                'dropout': [0.3, 0.3],
                'weight': 0.15,
                'samples_per_day': 48,
                'volatility_factor': 1.5
            },
            self.timeframe_keys['1h']: {
                'lstm_units': [128, 64],
                'dropout': [0.3, 0.3],
                'weight': 0.15,
                'samples_per_day': 24,
                'volatility_factor': 1.8
            },
            self.timeframe_keys['4h']: {
                'lstm_units': [256, 128, 64],
                'dropout': [0.4, 0.3, 0.3],
                'weight': 0.10,
                'samples_per_day': 6,
                'volatility_factor': 2.0
            },
            self.timeframe_keys['1d']: {
                'lstm_units': [256, 128, 64],
                'dropout': [0.4, 0.3, 0.3],
                'weight': 0.07,
                'samples_per_day': 1,
                'volatility_factor': 2.5
            },
            self.timeframe_keys['1w']: {
                'lstm_units': [256, 128, 64],
                'dropout': [0.4, 0.3, 0.3],
                'weight': 0.05,
                'samples_per_day': 1/7,
                'volatility_factor': 3.0
            },
            self.timeframe_keys['1M']: {
                'lstm_units': [256, 128, 64],
                'dropout': [0.4, 0.3, 0.3],
                'weight': 0.03,
                'samples_per_day': 1/30,
                'volatility_factor': 3.5
            }
        }
        
        # Configuration optimisée pour la mémoire
        self.memory_config = {
            'gru': {
                'units': [64, 32],
                'dropout': 0.3,
                'recurrent_dropout': 0.2,
                'stateful': True
            },
            'transformer': {
                'reduced_head_size': 32,
                'reduced_num_heads': 2,
                'max_sequence_length': 100,
                'chunk_size': 50
            }
        }
        
        # Configuration du Transformer
        self.transformer_config = {
            'base': {
                'head_size': 256,
                'num_heads': 4,
                'ff_dim_factor': 4,  # Multiplicateur de la dimension du feed-forward
                'num_blocks': 4,
                'mlp_units': [128],
                'dropout': 0.2,
                'mlp_dropout': 0.4,
                'attention_dropout': 0.1,
                'layer_norm_epsilon': 1e-6,
                'activation': 'gelu'  # ou 'relu'
            },
            # Configurations spécifiques par timeframe
            'timeframe_specific': {
                '5m': {
                    'num_blocks': 3,
                    'head_size': 128,
                    'num_heads': 4
                },
                '1h': {
                    'num_blocks': 4,
                    'head_size': 256,
                    'num_heads': 8
                },
                '1d': {
                    'num_blocks': 6,
                    'head_size': 512,
                    'num_heads': 8
                }
            }
        }
        
        # Initialiser le modèle
        if model_type == 'auto':
            print("\nMode auto: initialisation avec LSTM par défaut")
            self.model = self._build_lstm()
        else:
            self.model = self._build_transformer() if model_type == 'transformer' else self._build_lstm()
            
        if self.model:
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def _build_transformer(self):
        """
        Construit un modèle Transformer multi-timeframe avec configuration avancée
        
        Le modèle utilise:
        - Attention multi-tête avec dimension adaptative
        - Connexions résiduelles
        - Layer normalization
        - Feed-forward networks avec gating
        - Dropout adaptatif
        """
        print("\nConstruction du modèle Transformer multi-timeframe")
        
        inputs = {}
        transformer_outputs = []
        
        for tf in self.timeframe_config.keys():
            # Obtenir la configuration spécifique au timeframe ou utiliser la base
            config = self.transformer_config['timeframe_specific'].get(
                tf, self.transformer_config['base']
            )
            
            print(f"\nConfiguration Transformer pour {tf}:")
            print(f"• Nombre de blocks: {config['num_blocks']}")
            print(f"• Taille des têtes: {config['head_size']}")
            print(f"• Nombre de têtes: {config['num_heads']}")
            
            # Créer l'entrée
            input_name = f'input_{tf}'
            inputs[tf] = layers.Input(shape=self.input_shape, name=input_name)
            
            # Encodage positionnel
            x = self._add_positional_encoding(
                inputs[tf],
                max_length=self.input_shape[0],
                hidden_size=self.input_shape[1]
            )
            
            # Dropout initial
            x = layers.Dropout(config['dropout'])(x)
            
            # Blocks Transformer
            for i in range(config['num_blocks']):
                x = self._transformer_block(
                    x,
                    head_size=config['head_size'],
                    num_heads=config['num_heads'],
                    ff_dim=config['head_size'] * config['ff_dim_factor'],
                    dropout=config['dropout'],
                    attention_dropout=config['attention_dropout'],
                    epsilon=config['layer_norm_epsilon'],
                    activation=config['activation'],
                    block_name=f"{tf}_block_{i}"
                )
            
            # Pooling global avec attention
            x = self._attention_pooling(x, config['head_size'], name=f"{tf}_pool")
            transformer_outputs.append(x)
        
        # Fusion des sorties avec attention
        if len(transformer_outputs) > 1:
            x = self._fusion_layer(transformer_outputs)
        else:
            x = transformer_outputs[0]
        
        # MLP final avec residual connections
        for i, units in enumerate(self.transformer_config['base']['mlp_units']):
            residual = x
            x = layers.Dense(units, activation=None)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Activation(self.transformer_config['base']['activation'])(x)
            x = layers.Dropout(self.transformer_config['base']['mlp_dropout'])(x)
            if x.shape[-1] == residual.shape[-1]:
                x = layers.Add()([x, residual])
        
        # Sortie
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        
        return Model(inputs=list(inputs.values()), outputs=outputs)
    
    def _transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout, 
                         attention_dropout, epsilon, activation, block_name):
        """Block Transformer amélioré avec gating et connexions résiduelles"""
        
        # Multi-head attention avec pre-norm
        x = layers.LayerNormalization(epsilon=epsilon, name=f"{block_name}_norm1")(inputs)
        
        # Attention multi-tête avec masque causal
        attention_output = layers.MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            name=f"{block_name}_attention"
        )(x, x, x, use_causal_mask=True)
        
        # Residual connection
        x = layers.Add(name=f"{block_name}_add1")([attention_output, inputs])
        
        # Feed-forward network avec gating
        ff_norm = layers.LayerNormalization(epsilon=epsilon, name=f"{block_name}_norm2")(x)
        ff_inner = layers.Dense(ff_dim, activation=None, name=f"{block_name}_ff1")(ff_norm)
        
        # Gating mechanism
        gate = layers.Dense(ff_dim, activation='sigmoid', name=f"{block_name}_gate")(ff_norm)
        ff_gated = layers.Multiply(name=f"{block_name}_gating")([ff_inner, gate])
        
        ff_output = layers.Dense(inputs.shape[-1], name=f"{block_name}_ff2")(ff_gated)
        ff_output = layers.Dropout(dropout)(ff_output)
        
        return layers.Add(name=f"{block_name}_add2")([x, ff_output])
    
    def _attention_pooling(self, x, hidden_size, name):
        """Pooling avec attention pour réduire la séquence"""
        # Calculer les scores d'attention
        attention = layers.Dense(1, use_bias=False)(x)
        attention = layers.Softmax(axis=1)(attention)
        
        # Appliquer l'attention
        return layers.Dot(axes=(1, 1))([x, attention])
    
    def _fusion_layer(self, outputs):
        """Fusionne les sorties des différents timeframes avec attention"""
        # Concaténer
        concat = layers.Concatenate()(outputs)
        
        # Attention pour la fusion
        attention = layers.Dense(len(outputs), activation='softmax')(concat)
        
        # Pondérer et sommer
        weighted_sum = layers.Dot(axes=(1, 1))([
            layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs),
            attention
        ])
        
        return weighted_sum
    
    def _build_lstm(self):
        """
        Construit un modèle LSTM multi-timeframe avec configuration unifiée
        """
        print("\nConstruction du modèle LSTM multi-timeframe")
        
        # Utiliser les clés standardisées
        timeframes = list(self.timeframe_keys.values())
        
        # Créer les entrées pour chaque timeframe
        inputs = {}
        processed = {}
        
        for input_name in timeframes:
            if input_name not in self.timeframe_config:
                print(f"⚠️ Configuration manquante pour {input_name}, ignoré")
                continue
            
            config = self.timeframe_config[input_name]
            print(f"\nConfiguration pour {input_name}:")
            print(f"• LSTM units: {config['lstm_units']}")
            print(f"• Dropout rates: {config['dropout']}")
            print(f"• Poids: {config['weight']}")
            
            # Créer l'entrée avec le nom standardisé
            inputs[input_name] = layers.Input(shape=self.input_shape, name=input_name)
            
            # Construction des couches LSTM
            x = inputs[input_name]
            for i, (units, dropout) in enumerate(zip(config['lstm_units'], config['dropout'])):
                return_sequences = i < len(config['lstm_units']) - 1
                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i+1}_{input_name}'
                )(x)
                x = layers.Dropout(dropout, name=f'dropout_{i+1}_{input_name}')(x)
            
            processed[input_name] = x
        
        if not inputs:
            raise ValueError("Aucune entrée n'a été créée")
        
        # Concaténer toutes les sorties avec leurs poids respectifs
        weighted_outputs = []
        total_weight = 0
        
        for tf, output in processed.items():
            weight = self.timeframe_config[tf]['weight']
            weighted = layers.Lambda(
                lambda x: x * weight,
                name=f'weight_{tf}'
            )(output)
            weighted_outputs.append(weighted)
            total_weight += weight
        
        # Normaliser les poids
        if total_weight != 1.0:
            print(f"\n⚠️ Somme des poids ({total_weight}) != 1.0, normalisation appliquée")
            for i, tf in enumerate(processed.keys()):
                self.timeframe_config[tf]['weight'] /= total_weight
        
        concat = layers.Concatenate(name='concatenate')(weighted_outputs)
        
        # Couches denses finales
        x = layers.Dense(256, activation='relu', name='dense1')(concat)
        x = layers.Dropout(0.5, name='dropout_dense1')(x)
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.4, name='dropout_dense2')(x)
        outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
        
        # Créer le modèle
        model = Model(inputs=list(inputs.values()), outputs=outputs)
        
        # Debug des entrées
        print("\nConfiguration des entrées du modèle:")
        for input_tensor in model.inputs:
            print(f"• {input_tensor.name}: shape={input_tensor.shape}")
        
        return model
    
    def _positional_encoding(self, positions, d_model):
        """Encodage positionnel pour le Transformer"""
        
        angles = self._get_angles(
            positions=positions[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        angles[:, 0::2] = tf.sin(angles[:, 0::2])
        angles[:, 1::2] = tf.cos(angles[:, 1::2])
        
        pos_encoding = angles[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _get_angles(self, positions, i, d_model):
        """Calcule les angles pour l'encodage positionnel"""
        angle_rates = 1 / tf.pow(
            tf.maximum(
                tf.abs(angles), 
                1e-7
            )
        )  # Ajout de la parenthèse fermante manquante
        return positions * angle_rates
    
    def auto_select_model(self, X_train, y_train, X_val, y_val, test_epochs=5):
        """
        Teste les différentes architectures et sélectionne la meilleure
        """
        print("\nSélection automatique du modèle:")
        print("="*50)
        
        architectures = {
            'transformer': self._build_transformer,
            'lstm': self._build_lstm
        }
        
        best_model = None
        best_score = -float('inf')
        best_type = None
        
        for model_type, builder in architectures.items():
            print(f"\nTest de l'architecture {model_type.upper()}")
            
            # Construire et compiler le modèle
            test_model = builder()
            test_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            # Entraînement rapide
            history = test_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.batch_size,
                epochs=test_epochs,
                verbose=1
            )
            
            # Évaluation
            val_loss, val_acc = test_model.evaluate(X_val, y_val, verbose=0)
            
            # Prédictions pour métriques trading
            y_pred = np.argmax(test_model.predict(X_val), axis=1)
            
            # Calculer le score composite
            trading_metrics = self.calculate_trading_metrics(y_val, y_pred)
            composite_score = (
                0.3 * val_acc +
                0.3 * trading_metrics['win_rate'] +
                0.2 * (1 - val_loss) +
                0.2 * min(trading_metrics['sharpe_ratio'] / 3, 1)
            )
            
            print(f"\nScore composite: {composite_score:.3f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = test_model
                best_type = model_type
        
        # Mettre à jour le modèle avec la meilleure architecture
        self.model = best_model
        self.model_type = best_type
        
        print(f"\n✅ Meilleure architecture: {best_type.upper()}")
        print(f"Score final: {best_score:.3f}")
        
        return best_type, best_score
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        Entraîne le modèle
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation
            epochs (int): Nombre d'époques
        
        Returns:
            history: Historique d'entraînement
        """
        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=1
        )
    
    def predict(self, X: Dict) -> np.ndarray:
        """Prédit avec gestion robuste des timeframes manquants"""
        try:
            # Standardiser les clés des données d'entrée
            X_std = self.prepare_data(X)
            
            # Vérifier les timeframes manquants
            missing_timeframes = self._check_missing_timeframes(X_std)
            
            if missing_timeframes:
                print("\n⚠️ Timeframes manquants détectés:")
                for tf in missing_timeframes:
                    print(f"• {tf}")
                X_std = self._handle_missing_timeframes(X_std, missing_timeframes)
            
            return self.model.predict(X_std)
            
        except Exception as e:
            print(f"❌ Erreur dans predict: {str(e)}")
            raise

    def predict_with_confidence(self, X_dict, confidence_threshold=0.7):
        """
        Fait des prédictions avec seuil de confiance unifié par timeframe
        """
        predictions = self.model.predict(X_dict)
        
        # Ajuster les seuils selon la volatilité du timeframe
        adjusted_predictions = []
        for tf, config in self.timeframe_config.items():
            tf_predictions = predictions[list(X_dict.keys()).index(tf)]
            
            # Ajuster le seuil selon le facteur de volatilité
            adjusted_threshold = confidence_threshold * config['volatility_factor']
            
            # Appliquer le seuil ajusté
            mask = np.max(tf_predictions, axis=1) >= adjusted_threshold
            adjusted_preds = np.argmax(tf_predictions, axis=1)
            adjusted_preds[~mask] = 1  # Classe neutre pour les prédictions sous le seuil
            
            adjusted_predictions.append(adjusted_preds)
        
        # Combiner les prédictions avec les poids
        final_predictions = np.zeros_like(adjusted_predictions[0], dtype=float)
        total_weight = 0
        
        for i, tf in enumerate(self.timeframe_config.keys()):
            weight = self.timeframe_config[tf]['weight']
            final_predictions += adjusted_predictions[i] * weight
            total_weight += weight
        
        final_predictions /= total_weight
        
        return final_predictions

    def calculate_trading_metrics(self, y_true, y_pred, returns=None):
        """
        Calcul des métriques de trading améliorées
        
        Args:
            y_true (array): Labels réels
            y_pred (array): Prédictions
            returns (array, optional): Rendements pour le calcul du Sharpe
        
        Returns:
            dict: Métriques de trading
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'n_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        # Conversion des labels one-hot en classes si nécessaire
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Calcul des métriques de base
        n_trades = len(y_pred)
        correct_predictions = np.sum(y_true == y_pred)
        win_rate = correct_predictions / n_trades
        
        # Calcul du Sharpe Ratio si les returns sont fournis
        sharpe_ratio = 0
        if returns is not None and len(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Calcul du drawdown maximum
        if returns is not None and len(returns) > 0:
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0
        
        # Calcul du profit factor
        if returns is not None:
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            profit_factor = (np.sum(winning_trades) / -np.sum(losing_trades)) if np.sum(losing_trades) != 0 else 0
        else:
            profit_factor = 0
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    
    def _adjust_input_sizes(self, X: Dict) -> Dict:
        """Ajuste les tailles avec des clés standardisées"""
        try:
            # Standardiser les clés d'abord
            X_std = self.prepare_data(X)
            
            print("\n📊 Debug des tailles des timeframes:")
            for key, value in X_std.items():
                print(f"- {key}: shape={value.shape}")
            
            # Trouver la plus grande taille
            max_size = max(value.shape[0] for value in X_std.values())
            print(f"\nTaille maximale détectée: {max_size}")
            
            # Ajuster les tailles
            X_adjusted = {}
            for tf, data in X_std.items():
                if data.shape[0] != max_size:
                    if data.shape[0] == 1:  # C'est un placeholder
                        print(f"⚠️ Correction de {tf}, passage à {max_size} éléments")
                        X_adjusted[tf] = np.zeros((max_size, *data.shape[1:]))
                    else:  # Données réelles mais taille différente
                        print(f"⚠️ Ajustement de {tf} de {data.shape[0]} à {max_size} éléments")
                        if data.shape[0] > max_size:
                            X_adjusted[tf] = data[:max_size]
                        else:
                            # Répéter les données pour atteindre max_size
                            n_repeats = max_size // data.shape[0] + 1
                            repeated = np.tile(data, (n_repeats, 1, 1))
                            X_adjusted[tf] = repeated[:max_size]
                else:
                    X_adjusted[tf] = data
            
            print("\n📊 Tailles après ajustement:")
            for key, value in X_adjusted.items():
                print(f"- {key}: shape={value.shape}")
            
            return X_adjusted
            
        except Exception as e:
            print(f"❌ Erreur dans _adjust_input_sizes: {str(e)}")
            raise

    def calculate_risk_reward(self, y_true, y_pred, pip_target=25):
        """
        Calcule le ratio risque/récompense
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions
            pip_target (int): Objectif en pips
        
        Returns:
            float: Ratio risque/récompense
        """
        # Calculer les gains et pertes
        correct_trades = (y_true == y_pred) & (y_true != 0)
        incorrect_trades = (y_true != y_pred) & (y_pred != 0)
        
        n_correct = np.sum(correct_trades)
        n_incorrect = np.sum(incorrect_trades)
        
        if n_incorrect == 0:
            return float('inf')
        
        # Supposer que les trades gagnants atteignent l'objectif
        # et que les perdants sont stoppés à la moitié
        avg_gain = pip_target
        avg_loss = pip_target / 2
        
        risk_reward = (n_correct * avg_gain) / (n_incorrect * avg_loss)
        
        return risk_reward
    
    def summary(self):
        """Affiche le résumé du modèle"""
        self.model.summary()
    
    def _calculate_returns(self, predictions, prices):
        """Calcule les rendements réels basés sur les prix"""
        returns = np.zeros(len(predictions))
        
        for i in range(len(predictions)-1):
            if predictions[i] != 0:  # Si on prend une position
                # Calculer le P&L en pips
                if (predictions[i] == 1 and predictions[i+1] == 1) or (predictions[i] == -1 and predictions[i+1] == -1):
                    pnl_pips = 1  # Win = 1 pip
                else:
                    pnl_pips = -1  # Loss = -1 pip
                
                # Calculer le P&L en USD
                pnl_usd = pnl_pips * 0.1 * 100000
                
                # Mettre à jour la balance
                returns[i] = pnl_usd
        
        return returns
    
    def _calculate_max_streak(self, condition_array):
        """
        Calcule la plus longue séquence de True dans un array
        
        Args:
            condition_array: Array booléen à analyser
            
        Returns:
            int: Longueur de la plus longue séquence
        """
        if len(condition_array) == 0:
            return 0
            
        # Convertir en entiers (True -> 1, False -> 0)
        arr = condition_array.astype(int)
        
        # Trouver les changements
        changes = np.diff(np.hstack(([0], arr, [0])))
        
        # Indices où commence une séquence
        start_idx = np.where(changes == 1)[0]
        
        # Indices où finit une séquence
        end_idx = np.where(changes == -1)[0]
        
        if len(start_idx) == 0:  # Pas de séquence trouvée
            return 0
            
        # Calculer les longueurs des séquences
        streaks = end_idx - start_idx
        
        return np.max(streaks) if len(streaks) > 0 else 0

    def _calculate_transaction_costs(self, price: float, volatility: float) -> Dict:
        """Calcule les coûts de transaction dynamiques basés sur la volatilité"""
        try:
            # Coûts de base
            base_spread = 1.5  # 1.5 pips minimum
            base_slippage = 0.5  # 0.5 pips minimum
            
            # Ajustement selon la volatilité
            volatility_factor = max(1.0, volatility * 100)  # Normaliser la volatilité
            spread = min(base_spread * volatility_factor, 5.0)  # Max 5 pips
            slippage = min(base_slippage * volatility_factor, 2.0)  # Max 2 pips
            
            # Conversion en points de prix
            pip_value = 0.01  # 1 pip = 0.01 pour XAU/USD
            spread_points = spread * pip_value
            slippage_points = slippage * pip_value
            
            return {
                'spread': spread_points,
                'slippage': slippage_points,
                'total_cost': spread_points + slippage_points
            }
        except Exception as e:
            print(f"❌ Erreur dans _calculate_transaction_costs: {str(e)}")
            return {'spread': 0.015, 'slippage': 0.005, 'total_cost': 0.02}

    def _calculate_position_pnl(self, position: Dict, current_price: float) -> float:
        """Calcule le P&L d'une position avec coûts de transaction"""
        try:
            direction = 1 if position['type'] == 'long' else -1
            price_diff = (current_price - position['entry_price']) * direction
            
            # Appliquer les coûts de transaction
            total_cost = position['costs']['total_cost']
            
            # Calculer le P&L net
            pnl = (price_diff - total_cost) * position['size']
            
            return pnl
            
        except Exception as e:
            print(f"❌ Erreur dans _calculate_position_pnl: {str(e)}")
            return 0.0

    def _calculate_position_size(self, balance: float, price: float, volatility: float) -> float:
        """Calcule la taille de position adaptée au risque"""
        try:
            # Risque de base (2% du capital)
            risk_amount = balance * 0.02
            
            # Ajuster selon la volatilité
            volatility_factor = max(0.5, min(2.0, 1.0 / volatility))
            adjusted_risk = risk_amount * volatility_factor
            
            # Calculer la taille en lots
            position_size = adjusted_risk / (price * 100)  # 1 lot = 100 unités
            
            # Limites de taille
            min_size = 0.01  # 0.01 lot minimum
            max_size = 1.0   # 1 lot maximum
            
            return np.clip(position_size, min_size, max_size)
            
        except Exception as e:
            print(f"❌ Erreur dans _calculate_position_size: {str(e)}")
            return 0.01

    def backtest_model(self, X_test_dict, y_test, confidence_threshold=0.7, 
                      pip_threshold=0.0025, initial_balance=100000):
        """Backtest avec coûts de transaction réalistes"""
        try:
            print("\n🔄 Démarrage du backtest multi-timeframe")
            
            # 1. Obtenir les prédictions pour chaque timeframe
            predictions_by_tf = {}
            for tf, X_test in X_test_dict.items():
                print(f"\nPrédictions pour {tf}:")
                predictions = self.predict_with_confidence(
                    {tf: X_test},
                    confidence_threshold=confidence_threshold
                )
                predictions_by_tf[tf] = predictions
                print(f"• Shape: {predictions.shape}")
            
            # 2. Fusionner les prédictions avec pondération par timeframe
            print("\nFusion des prédictions:")
            final_predictions = np.zeros_like(predictions_by_tf['input_5m'])  # Utiliser 5m comme référence
            total_weight = 0
            
            for tf, predictions in predictions_by_tf.items():
                # Ajuster les prédictions à la taille de référence si nécessaire
                if predictions.shape[0] != final_predictions.shape[0]:
                    print(f"⚠️ Redimensionnement {tf}: {predictions.shape[0]} → {final_predictions.shape[0]}")
                    if predictions.shape[0] > final_predictions.shape[0]:
                        predictions = predictions[:final_predictions.shape[0]]
                    else:
                        # Répéter les dernières prédictions
                        pad_length = final_predictions.shape[0] - predictions.shape[0]
                        predictions = np.pad(predictions, (0, pad_length), 'edge')
                
                # Appliquer le poids du timeframe
                weight = self.timeframe_config[tf]['weight']
                final_predictions += predictions * weight
                total_weight += weight
                print(f"• {tf}: poids = {weight:.2f}")
            
            # Normaliser les prédictions
            final_predictions /= total_weight
            
            # 3. Récupérer les prix du timeframe de référence
            reference_tf = 'input_5m'
            normalized_prices = X_test_dict[reference_tf][:, -1, 3]  # [batch, dernière_step, close_price]
            
            # 4. Dénormaliser les prix
            try:
                prices = self.transform_prices(normalized_prices, inverse=True)
                print("\n📊 Vérification des prix:")
                print(f"• Prix moyen: {np.mean(prices):.2f}")
                print(f"• Prix min: {np.min(prices):.2f}")
                print(f"• Prix max: {np.max(prices):.2f}")
                
                # Vérifier la cohérence des prix
                scaler_config = self.price_scalers[self.current_instrument]
                if not (scaler_config['min_price'] <= np.mean(prices) <= scaler_config['max_price']):
                    print("\n⚠️ Prix dénormalisés hors plage attendue")
                    print(f"• Prix moyen: {np.mean(prices):.2f}")
                    print(f"• Plage attendue: [{scaler_config['min_price']}, {scaler_config['max_price']}]")
                    
                    # Correction si nécessaire
                    if np.mean(prices) < scaler_config['min_price']:
                        multiplier = 10 ** int(np.log10(scaler_config['base_price'] / np.mean(prices)))
                        prices *= multiplier
                        print(f"• Application d'un multiplicateur x{multiplier}")
                        print(f"• Nouveaux prix moyens: {np.mean(prices):.2f}")
            
            except Exception as e:
                print(f"\n❌ Erreur lors de la dénormalisation des prix: {str(e)}")
                print("Utilisation des prix normalisés avec base de référence")
                base_price = self.price_scalers[self.current_instrument]['base_price']
                prices = base_price * (1 + normalized_prices)
            
            # Calcul de la volatilité glissante
            volatility = pd.Series(prices).pct_change().rolling(20).std()
            
            # 5. Simuler le trading
            balance = initial_balance
            positions = []
            equity_curve = [balance]
            trades = []
            current_position = None
            
            print("\n📈 Simulation du trading:")
            for i in range(1, len(prices)):
                current_vol = volatility.iloc[i-1] if i > 20 else volatility.mean()
                
                # Gérer la position existante
                if current_position is not None:
                    pnl = self._calculate_position_pnl(current_position, prices[i])
                    
                    # Vérifier les conditions de sortie
                    if self._should_close_position(current_position, prices[i], pnl):
                        # Ajouter les coûts de sortie
                        exit_costs = self._calculate_transaction_costs(prices[i], current_vol)
                        pnl -= exit_costs['total_cost'] * current_position['size']
                        
                        balance += pnl
                        trades.append({
                            'entry_price': current_position['entry_price'],
                            'exit_price': prices[i],
                            'pnl': pnl,
                            'costs': current_position['costs'],
                            'exit_costs': exit_costs,
                            'duration': i - current_position['entry_index'],
                            'type': current_position['type']
                        })
                        current_position = None
                
                # Vérifier le signal d'entrée
                if current_position is None:
                    signal = np.argmax(final_predictions[i]) - 1
                    confidence = np.max(final_predictions[i])
                    
                    if abs(signal) > 0 and confidence >= confidence_threshold:
                        # Calculer les coûts de transaction
                        entry_costs = self._calculate_transaction_costs(prices[i], current_vol)
                        
                        # Calculer la taille de position
                        position_size = self._calculate_position_size(
                            balance, prices[i], current_vol
                        )
                        
                        current_position = {
                            'type': 'long' if signal > 0 else 'short',
                            'entry_price': prices[i],
                            'size': position_size,
                            'entry_index': i,
                            'costs': entry_costs,
                            'volatility': current_vol
                        }
                        
                        print(f"\nOuverture position {current_position['type']}:")
                        print(f"• Prix: {prices[i]:.2f}")
                        print(f"• Taille: {position_size:.2f}")
                        print(f"• Coûts: {entry_costs['total_cost']:.4f}")
                        print(f"• Confiance: {confidence:.1%}")
                
                # Mettre à jour l'equity curve
                equity_curve.append(balance)
            
            # Analyse des trades
            if trades:
                costs_analysis = self._analyze_trading_costs(trades)
                print("\n💰 Analyse des coûts:")
                print(f"• Coût moyen par trade: {costs_analysis['avg_cost']:.4f}")
                print(f"• Impact sur le P&L: {costs_analysis['cost_impact']:.1%}")
                print(f"• Ratio coût/profit: {costs_analysis['cost_profit_ratio']:.2f}")
            
            # 6. Calculer les métriques finales
            roi = (balance - initial_balance) / initial_balance
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
            
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            results = {
                'roi': roi,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'final_balance': balance,
                'equity_curve': equity_curve,
                'trades': trades
            }
            
            print("\n📊 Résultats du backtest:")
            print(f"• ROI: {roi:.1%}")
            print(f"• Nombre de trades: {len(trades)}")
            print(f"• Win rate: {win_rate:.1%}")
            print(f"• Drawdown max: {max_drawdown:.1%}")
            print(f"• Ratio de Sharpe: {sharpe_ratio:.2f}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Erreur lors du backtest: {str(e)}")
            raise

    def _analyze_trading_costs(self, trades: List[Dict]) -> Dict:
        """Analyse l'impact des coûts de transaction"""
        try:
            total_costs = sum(
                t['costs']['total_cost'] + t['exit_costs']['total_cost'] 
                for t in trades
            )
            total_pnl = sum(t['pnl'] for t in trades)
            
            return {
                'avg_cost': total_costs / len(trades),
                'cost_impact': abs(total_costs / total_pnl) if total_pnl != 0 else float('inf'),
                'cost_profit_ratio': total_costs / sum(t['pnl'] for t in trades if t['pnl'] > 0)
                if sum(t['pnl'] for t in trades if t['pnl'] > 0) > 0 else float('inf')
            }
            
        except Exception as e:
            print(f"❌ Erreur dans _analyze_trading_costs: {str(e)}")
            return {'avg_cost': 0, 'cost_impact': 0, 'cost_profit_ratio': 0}

    def train_multi_timeframe(self, X_train, y_train, validation_data=None, epochs=50, 
                             batch_size=32, verbose=1, callbacks=None):
        """
        Entraîne le modèle sur des données multi-timeframes
        
        Args:
            X_train: Dict de données d'entraînement par timeframe
            y_train: Labels d'entraînement
            validation_data: Tuple (X_val, y_val) pour la validation
            epochs: Nombre d'époques
            batch_size: Taille du batch
            verbose: Niveau de verbosité
            callbacks: Liste de callbacks Keras
            
        Returns:
            Historique d'entraînement
        """
        try:
            # Préparation des données de validation
            validation_split = None
            if validation_data is not None:
                X_val, y_val = validation_data
                validation_split = (X_val, y_val)
            
            # Configuration des callbacks par défaut si non fournis
            if callbacks is None:
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        min_delta=0.0001
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=5,
                        min_lr=1e-7
                    )
                ]
            
            # Entraînement du modèle
            history = self.fit(
                X_train,
                y_train,
                validation_data=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            return history
            
        except Exception as e:
            print(f"Erreur dans train_multi_timeframe: {str(e)}")
            raise

    def _data_generator(self, X, y, batch_size):
        """Générateur de données pour l'entraînement par lots"""
        try:
            while True:
                # Indices pour le batch courant
                indices = np.random.permutation(len(y))
                
                for start in range(0, len(y), batch_size):
                    end = min(start + batch_size, len(y))
                    batch_indices = indices[start:end]
                    
                    # Préparer le batch
                    X_batch = {
                        k: v[batch_indices] 
                        for k, v in X.items()
                    }
                    y_batch = y[batch_indices]
                    
                    yield X_batch, y_batch
                    
        except Exception as e:
            print(f"❌ Erreur dans _data_generator: {str(e)}")
            raise

    def _build_memory_efficient_model(self):
        """Construit un modèle optimisé pour la mémoire"""
        try:
            print("\n🔧 Construction du modèle optimisé pour la mémoire")
            
            # Utiliser GRU au lieu de LSTM/Transformer
            inputs = {}
            processed = {}
            
            for tf_key in self.timeframe_keys.values():
                if tf_key not in self.timeframe_config:
                    continue
                
                # Créer l'entrée
                inputs[tf_key] = layers.Input(shape=self.input_shape, name=tf_key)
                
                # GRU avec gestion optimisée de la mémoire
                x = self._build_gru_branch(inputs[tf_key], tf_key)
                processed[tf_key] = x
            
            # Fusion optimisée des timeframes
            combined = self._merge_timeframes_efficiently(processed)
            
            # Couches de sortie réduites
            x = layers.Dense(32, activation='relu')(combined)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.n_classes, activation='softmax')(x)
            
            model = Model(inputs=list(inputs.values()), outputs=outputs)
            
            # Optimiseur avec gestion mémoire
            optimizer = self._get_memory_efficient_optimizer()
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"❌ Erreur dans _build_memory_efficient_model: {str(e)}")
            raise

    def _build_gru_branch(self, inputs, timeframe):
        """Construit une branche GRU optimisée"""
        try:
            config = self.memory_config['gru']
            
            x = inputs
            for i, units in enumerate(config['units']):
                return_sequences = i < len(config['units']) - 1
                
                x = layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=config['dropout'],
                    recurrent_dropout=config['recurrent_dropout'],
                    stateful=config['stateful'],
                    name=f'gru_{i+1}_{timeframe}'
                )(x)
            
            return x
            
        except Exception as e:
            print(f"❌ Erreur dans _build_gru_branch: {str(e)}")
            raise

    def _merge_timeframes_efficiently(self, processed_inputs):
        """Fusionne les timeframes de manière optimisée"""
        try:
            weighted_outputs = []
            total_weight = 0
            
            for tf, output in processed_inputs.items():
                weight = self.timeframe_config[tf]['weight']
                
                # Réduction de dimension avant la pondération
                reduced = layers.Dense(32, activation='relu')(output)
                weighted = layers.Lambda(lambda x: x * weight)(reduced)
                
                weighted_outputs.append(weighted)
                total_weight += weight
            
            # Fusion progressive pour économiser la mémoire
            merged = weighted_outputs[0]
            for output in weighted_outputs[1:]:
                merged = layers.Add()([merged, output])
            
            return layers.Lambda(lambda x: x / total_weight)(merged)
            
        except Exception as e:
            print(f"❌ Erreur dans _merge_timeframes_efficiently: {str(e)}")
            raise

    def _get_memory_efficient_optimizer(self):
        """Configure un optimiseur économe en mémoire"""
        try:
            return Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,  # Désactiver AMSGrad pour économiser la mémoire
                clipnorm=1.0    # Limiter les gradients
            )
        except Exception as e:
            print(f"❌ Erreur dans _get_memory_efficient_optimizer: {str(e)}")
            return Adam(learning_rate=self.learning_rate)

    def save(self, path):
        """
        Sauvegarde le modèle et ses paramètres
        
        Args:
            path (str): Chemin de sauvegarde (sans extension)
        """
        try:
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Sauvegarder le modèle Keras
            self.model.save(f"{path}.h5")
            
            # Sauvegarder les paramètres
            params = {
                'input_shape': self.input_shape,
                'n_classes': self.n_classes,
                'model_type': self.model_type,
                'learning_rate': self.learning_rate,
                'lstm_units': self.lstm_units,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate,
                'scaler_params': {
                    'mean_': self.feature_scaler.mean_.tolist() if hasattr(self.feature_scaler, 'mean_') else None,
                    'scale_': self.feature_scaler.scale_.tolist() if hasattr(self.feature_scaler, 'scale_') else None,
                }
            }
            
            with open(f"{path}_params.json", 'w') as f:
                json.dump(params, f, indent=4)
            
            print(f"\n✅ Modèle sauvegardé: {path}")
            print(f"  • Modèle: {path}.h5")
            print(f"  • Paramètres: {path}_params.json")
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la sauvegarde du modèle: {str(e)}")
            raise

    def fit_price_scaler(self, prices, instrument='XAUUSD'):
        """
        Ajuste le scaler de prix pour un instrument donné
        
        Args:
            prices (array-like): Prix à utiliser pour l'ajustement
            instrument (str): Identifiant de l'instrument
        """
        if instrument not in self.price_scalers:
            raise ValueError(f"Instrument non supporté: {instrument}")
        
        scaler_config = self.price_scalers[instrument]
        prices = np.array(prices).reshape(-1, 1)
        
        # Vérifier si les prix sont dans la plage attendue
        mean_price = np.mean(prices)
        if mean_price < scaler_config['min_price'] or mean_price > scaler_config['max_price']:
            print(f"\n⚠️ Attention: Prix moyens ({mean_price:.2f}) hors plage attendue")
            print(f"Plage attendue: [{scaler_config['min_price']}, {scaler_config['max_price']}]")
            
            # Ajuster les prix si nécessaire
            if mean_price < scaler_config['min_price']:
                multiplier = 10 ** int(np.log10(scaler_config['base_price'] / mean_price))
                print(f"Application d'un multiplicateur x{multiplier}")
                prices *= multiplier
        
        # Ajuster le scaler
        scaler_config['scaler'].fit(prices)
        print(f"\nScaler de prix ajusté pour {instrument}")
        print(f"• Moyenne: {scaler_config['scaler'].mean_[0]:.2f}")
        print(f"• Écart-type: {scaler_config['scaler'].scale_[0]:.2f}")
    
    def transform_prices(self, prices, instrument='XAUUSD', inverse=False):
        """
        Transforme les prix (normalisation ou dénormalisation)
        
        Args:
            prices (array-like): Prix à transformer
            instrument (str): Identifiant de l'instrument
            inverse (bool): Si True, effectue une transformation inverse
            
        Returns:
            array: Prix transformés
        """
        if instrument not in self.price_scalers:
            raise ValueError(f"Instrument non supporté: {instrument}")
        
        scaler = self.price_scalers[instrument]['scaler']
        prices = np.array(prices).reshape(-1, 1)
        
        if inverse:
            return scaler.inverse_transform(prices).flatten()
        else:
            return scaler.transform(prices).flatten()

    def fit(self, X, y, validation_data=None, validation_split=0.2, epochs=50, batch_size=32, 
            verbose=1, callbacks=None):
        """
        Entraîne le modèle sur les données multi-timeframe
        
        Args:
            X: Dict de données d'entrée par timeframe {'input_tf': array}
            y: Labels d'entraînement
            validation_data: Tuple (X_val, y_val) pour la validation
            validation_split: Proportion des données pour la validation
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            verbose: Niveau de verbosité
            callbacks: Liste de callbacks Keras
            
        Returns:
            history: Historique d'entraînement
        """
        try:
            print("\nPréparation des données pour l'entraînement multi-timeframe")
            
            # Vérifier la cohérence des données
            if not isinstance(X, dict):
                raise ValueError("X doit être un dictionnaire de données par timeframe")
            
            print("\nStructure des données:")
            for tf, data in X.items():
                print(f"• {tf}: shape={data.shape}")
            print(f"• Labels: shape={y.shape}")
            
            # Utiliser directement model.fit avec tous les paramètres
            return self.model.fit(
                X,
                y,
                validation_data=validation_data,
                validation_split=validation_split if validation_data is None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'entraînement: {str(e)}")
            print("\nDétails des données:")
            print("X keys:", list(X.keys()))
            if isinstance(y, dict):
                print("y keys:", list(y.keys()))
            raise

    def notify_order(self, order):
        """Gestion des notifications d'ordres"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'ACHAT EXÉCUTÉ, Prix: {order.executed.price:.2f}')
            else:
                self.log(f'VENTE EXÉCUTÉE, Prix: {order.executed.price:.2f}')
                
            # Mettre à jour les trades
            if order.exectype == bt.Order.Market:
                self._update_trade_info(order)

    def prepare_data(self, data: Dict) -> Dict:
        """Standardise les clés des données d'entrée"""
        try:
            standardized_data = {}
            
            # Conversion des clés vers le format standardisé
            for tf, tf_key in self.timeframe_keys.items():
                if tf in data:
                    standardized_data[tf_key] = data[tf]
                elif tf_key in data:
                    standardized_data[tf_key] = data[tf_key]
            
            if not standardized_data:
                raise ValueError("Aucune donnée valide après standardisation des clés")
            
            print("\nDonnées standardisées:")
            for key, value in standardized_data.items():
                print(f"• {key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
            
            return standardized_data
            
        except Exception as e:
            print(f"❌ Erreur dans prepare_data: {str(e)}")
            raise

    def _check_missing_timeframes(self, X: Dict) -> List[str]:
        """Vérifie les timeframes manquants"""
        try:
            expected_timeframes = set(self.timeframe_keys.values())
            provided_timeframes = set(X.keys())
            
            return list(expected_timeframes - provided_timeframes)
            
        except Exception as e:
            print(f"❌ Erreur dans _check_missing_timeframes: {str(e)}")
            return []

    def _handle_missing_timeframes(self, X: Dict, missing_timeframes: List[str]) -> Dict:
        """Gère les timeframes manquants de manière dynamique"""
        try:
            X_complete = X.copy()
            
            if not X:
                raise ValueError("Aucune donnée disponible pour générer les timeframes manquants")
            
            # Récupérer un exemple de shape pour les données manquantes
            sample_shape = next(iter(X.values())).shape
            
            for missing_tf in missing_timeframes:
                print(f"\nGénération des données pour {missing_tf}:")
                
                # 1. Essayer l'interpolation à partir des timeframes proches
                interpolated_data = self._interpolate_from_neighbors(
                    X, missing_tf, sample_shape
                )
                
                if interpolated_data is not None:
                    X_complete[missing_tf] = interpolated_data
                    print("✓ Données générées par interpolation")
                    continue
                
                # 2. Si l'interpolation échoue, utiliser la moyenne pondérée
                weighted_data = self._generate_weighted_data(
                    X, missing_tf, sample_shape
                )
                
                if weighted_data is not None:
                    X_complete[missing_tf] = weighted_data
                    print("✓ Données générées par moyenne pondérée")
                    continue
                
                # 3. En dernier recours, utiliser des valeurs par défaut
                print("⚠️ Utilisation des valeurs par défaut")
                X_complete[missing_tf] = self._generate_default_data(sample_shape)
            
            return X_complete
            
        except Exception as e:
            print(f"❌ Erreur dans _handle_missing_timeframes: {str(e)}")
            return X

    def _interpolate_from_neighbors(self, X: Dict, missing_tf: str, 
                                  target_shape: tuple) -> np.ndarray:
        """Interpole les données à partir des timeframes voisins"""
        try:
            # Trouver les timeframes voisins
            tf_order = list(self.timeframe_keys.values())
            missing_idx = tf_order.index(missing_tf)
            
            lower_tf = tf_order[missing_idx - 1] if missing_idx > 0 else None
            upper_tf = tf_order[missing_idx + 1] if missing_idx < len(tf_order) - 1 else None
            
            if lower_tf in X and upper_tf in X:
                # Interpolation entre les deux timeframes voisins
                lower_data = X[lower_tf]
                upper_data = X[upper_tf]
                
                # Calculer les poids selon la distance temporelle
                lower_weight = self._get_timeframe_weight(lower_tf, missing_tf)
                upper_weight = self._get_timeframe_weight(upper_tf, missing_tf)
                total_weight = lower_weight + upper_weight
                
                # Interpolation pondérée
                interpolated = (
                    lower_data * (lower_weight / total_weight) +
                    upper_data * (upper_weight / total_weight)
                )
                
                return interpolated
                
            return None
            
        except Exception as e:
            print(f"❌ Erreur dans _interpolate_from_neighbors: {str(e)}")
            return None

    def _generate_weighted_data(self, X: Dict, missing_tf: str, 
                              target_shape: tuple) -> np.ndarray:
        """Génère des données par moyenne pondérée des timeframes disponibles"""
        try:
            weighted_sum = np.zeros(target_shape)
            total_weight = 0
            
            for tf, data in X.items():
                # Calculer le poids selon la "distance" temporelle
                weight = self._get_timeframe_weight(tf, missing_tf)
                weighted_sum += data * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
                
            return None
            
        except Exception as e:
            print(f"❌ Erreur dans _generate_weighted_data: {str(e)}")
            return None

    def _get_timeframe_weight(self, source_tf: str, target_tf: str) -> float:
        """Calcule le poids selon la distance temporelle entre timeframes"""
        try:
            # Convertir les timeframes en minutes
            tf_minutes = {
                'input_5m': 5,
                'input_15m': 15,
                'input_30m': 30,
                'input_1h': 60,
                'input_4h': 240,
                'input_1d': 1440,
                'input_1w': 10080,
                'input_1M': 43200
            }
            
            source_minutes = tf_minutes[source_tf]
            target_minutes = tf_minutes[target_tf]
            
            # Poids inversement proportionnel à la différence logarithmique
            time_diff = abs(np.log(source_minutes) - np.log(target_minutes))
            weight = 1 / (1 + time_diff)
            
            return weight
            
        except Exception as e:
            print(f"❌ Erreur dans _get_timeframe_weight: {str(e)}")
            return 0.0

    def _generate_default_data(self, shape: tuple) -> np.ndarray:
        """Génère des données par défaut basées sur la distribution typique"""
        try:
            # Générer des données avec une distribution normale
            mean = 0.0
            std = 0.1  # Écart-type réduit pour des valeurs conservatrices
            
            default_data = np.random.normal(mean, std, shape)
            
            # Appliquer une structure temporelle simple
            if len(shape) > 2:  # Pour les séquences temporelles
                # Ajouter une tendance légère
                time_steps = np.linspace(0, 0.1, shape[1])
                trend = time_steps.reshape(1, -1, 1)
                default_data += trend
            
            return default_data
            
        except Exception as e:
            print(f"❌ Erreur dans _generate_default_data: {str(e)}")
            return np.zeros(shape)

class MLTradingStrategy(bt.Strategy):
    """Stratégie Backtrader utilisant le modèle ML"""
    
    params = (
        ('model', None),
        ('confidence_threshold', 0.8),
        ('stop_loss', 0.02),        # 2%
        ('take_profit', 0.03),      # 3%
        ('position_size', 0.02),    # 2% du capital par trade
        ('min_volatility', 0.001),  # Seuil minimum de volatilité
        ('max_positions', 3),       # Nombre maximum de positions simultanées
        ('timeframes', ['5m', '15m', '30m', '1h', '4h']),  # Timeframes à utiliser
        ('sequence_length', 60),    # Longueur des séquences pour le modèle
    )
    
    def __init__(self):
        """Initialisation de la stratégie"""
        super().__init__()
        
        self.model = self.params.model
        if not self.model:
            raise ValueError("Modèle ML non fourni")
        
        # Ordres et positions
        self.orders = {}  # {data_name: Order}
        self.stops = {}   # {data_name: Stop order}
        self.targets = {} # {data_name: Target order}
        self.positions = {} # {data_name: Position info}
        
        # Historique des trades
        self.trades = []
        self.current_trade = None
        
        # Buffer pour les features
        self.feature_buffers = {tf: [] for tf in self.params.timeframes}
        
        # Indicateurs techniques
        self._init_indicators()
        
        print("\n🔧 Initialisation de la stratégie:")
        print(f"• Seuil de confiance: {self.params.confidence_threshold}")
        print(f"• Stop loss: {self.params.stop_loss*100}%")
        print(f"• Take profit: {self.params.take_profit*100}%")
        print(f"• Taille position: {self.params.position_size*100}%")
    
    def _init_indicators(self):
        """Initialise les indicateurs techniques"""
        for data in self.datas:
            # Indicateurs de base
            setattr(self, f'returns_{data._name}', 
                   bt.indicators.Returns(data.close, period=1))
            setattr(self, f'log_returns_{data._name}', 
                   bt.indicators.LogReturns(data.close, period=1))
            
            # Volatilité
            setattr(self, f'volatility_20_{data._name}',
                   bt.indicators.StdDev(getattr(self, f'log_returns_{data._name}'), period=20))
            setattr(self, f'volatility_50_{data._name}',
                   bt.indicators.StdDev(getattr(self, f'log_returns_{data._name}'), period=50))
            
            # Moyennes mobiles
            setattr(self, f'sma_20_{data._name}',
                   bt.indicators.SMA(data.close, period=20))
            setattr(self, f'sma_50_{data._name}',
                   bt.indicators.SMA(data.close, period=50))
            
            # RSI
            setattr(self, f'rsi_{data._name}',
                   bt.indicators.RSI(data.close, period=14))
    
    def next(self):
        """Logique principale de trading"""
        for data in self.datas:
            self._process_data(data)
    
    def _process_data(self, data):
        """Traite les données pour un timeframe spécifique"""
        try:
            # 1. Mettre à jour le buffer de features
            features = self._prepare_features(data)
            self.feature_buffers[data._name].append(features)
            
            # Garder seulement la longueur nécessaire
            if len(self.feature_buffers[data._name]) > self.params.sequence_length:
                self.feature_buffers[data._name].pop(0)
            
            # 2. Vérifier si nous avons assez de données
            if len(self.feature_buffers[data._name]) < self.params.sequence_length:
                return
            
            # 3. Préparer les données pour le modèle
            X = self._prepare_model_input(data)
            
            # 4. Obtenir la prédiction
            prediction = self.model.predict_with_confidence(
                {data._name: X},
                confidence_threshold=self.params.confidence_threshold
            )
            
            # 5. Vérifier les conditions de trading
            if self._check_trading_conditions(data, prediction):
                self._execute_trading_decision(data, prediction)
            
            # 6. Gérer les positions existantes
            self._manage_positions(data)
            
        except Exception as e:
            self.log(f"Erreur dans _process_data: {str(e)}")
    
    def _prepare_features(self, data):
        """Prépare les features pour une barre"""
        return np.array([
            data.close[0],
            data.high[0],
            data.low[0],
            data.volume[0],
            getattr(self, f'returns_{data._name}')[0],
            getattr(self, f'volatility_20_{data._name}')[0],
            getattr(self, f'volatility_50_{data._name}')[0],
            getattr(self, f'sma_20_{data._name}')[0],
            getattr(self, f'sma_50_{data._name}')[0],
            getattr(self, f'rsi_{data._name}')[0]
        ])
    
    def _prepare_model_input(self, data):
        """Prépare les données pour le modèle"""
        features = np.array(self.feature_buffers[data._name])
        return features.reshape(1, self.params.sequence_length, -1)
    
    def _check_trading_conditions(self, data, prediction):
        """Vérifie les conditions pour trader"""
        # 1. Vérifier la volatilité
        current_vol = getattr(self, f'volatility_20_{data._name}')[0]
        if current_vol < self.params.min_volatility:
            return False
        
        # 2. Vérifier le nombre de positions
        if len(self.positions) >= self.params.max_positions:
            return False
        
        # 3. Vérifier si nous avons déjà une position sur cet actif
        if data._name in self.positions:
            return False
        
        # 4. Vérifier le RSI (éviter les extrêmes)
        rsi = getattr(self, f'rsi_{data._name}')[0]
        if rsi > 70 or rsi < 30:
            return False
        
        return True
    
    def _execute_trading_decision(self, data, prediction):
        """Exécute la décision de trading"""
        if prediction[0] == 2:  # Signal d'achat
            self._enter_long(data)
        elif prediction[0] == 0:  # Signal de vente
            self._enter_short(data)
    
    def _enter_long(self, data):
        """Entre en position longue"""
        if data._name in self.orders:
            return
        
        # Calculer la taille de la position
        size = self._calculate_position_size(data)
        
        # Placer l'ordre principal
        price = data.close[0]
        self.orders[data._name] = self.buy(
            data=data,
            size=size,
            exectype=bt.Order.Market
        )
        
        # Placer les ordres de stop et target
        stop_price = price * (1 - self.params.stop_loss)
        target_price = price * (1 + self.params.take_profit)
        
        self.stops[data._name] = self.sell(
            data=data,
            size=size,
            exectype=bt.Order.Stop,
            price=stop_price,
            parent=self.orders[data._name]
        )
        
        self.targets[data._name] = self.sell(
            data=data,
            size=size,
            exectype=bt.Order.Limit,
            price=target_price,
            parent=self.orders[data._name]
        )
        
        self._log_trade_entry('LONG', data, price, size)
    
    def _enter_short(self, data):
        """Entre en position courte"""
        # Code similaire à _enter_long mais pour les positions courtes
        pass
    
    def _manage_positions(self, data):
        """Gère les positions existantes"""
        if data._name not in self.positions:
            return
        
        position = self.positions[data._name]
        current_price = data.close[0]
        
        # Vérifier les conditions de sortie dynamiques
        if self._check_exit_conditions(data, position):
            self._close_position(data)
    
    def _calculate_position_size(self, data):
        """Calcule la taille de la position"""
        cash = self.broker.getcash()
        position_value = cash * self.params.position_size
        return position_value / data.close[0]
    
    def notify_order(self, order):
        """Gestion des notifications d'ordres"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'ACHAT EXÉCUTÉ, Prix: {order.executed.price:.2f}')
            else:
                self.log(f'VENTE EXÉCUTÉE, Prix: {order.executed.price:.2f}')
                
            # Mettre à jour les trades
            if order.exectype == bt.Order.Market:
                self._update_trade_info(order)
    
    def log(self, txt, dt=None):
        """Logging avec timestamp"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

def backtest_model(self, data, initial_capital=100000.0):
    """
    Effectue un backtest du modèle
    
    Args:
        data: DataFrame avec les données OHLCV
        initial_capital: Capital initial pour le backtest
    
    Returns:
        dict: Résultats du backtest
    """
    # Créer le cerveau Backtrader
    cerebro = bt.Cerebro()
    
    # Ajouter la stratégie
    cerebro.addstrategy(
        MLTradingStrategy,
        model=self,
        confidence_threshold=0.8
    )
    
    # Préparer les données
    data_feed = self._prepare_backtrader_data(data)
    cerebro.adddata(data_feed)
    
    # Configurer le broker
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # Ajouter des analyseurs
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Exécuter le backtest
    print("\nDémarrage du backtest:")
    print("="*50)
    
    results = cerebro.run()
    strategy = results[0]
    
    # Analyser les résultats
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_capital
    roi = (pnl / initial_capital) * 100
    
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    
    # Debug du drawdown
    print("\n🔍 Debug Drawdown:")
    print(f"• Drawdown actuel: {drawdown.get('drawdown', 0):.2f}%")
    print(f"• Drawdown maximum: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    print(f"• Durée du drawdown max: {drawdown.get('max', {}).get('len', 0)} périodes")
    print(f"• Monnaie perdue en drawdown: ${drawdown.get('max', {}).get('moneydown', 0):,.2f}")
    
    # Debug des trades
    print("\n🔍 Debug des Trades:")
    print(f"• Nombre total de trades: {trades.total.total if hasattr(trades, 'total') else 0}")
    print(f"• Trades gagnants: {trades.won.total if hasattr(trades, 'won') else 0}")
    print(f"• Trades perdants: {trades.lost.total if hasattr(trades, 'lost') else 0}")
    
    # Debug de la gestion du risque
    print("\n🔍 Debug Gestion du Risque:")
    print(f"• Capital initial: ${initial_capital:,.2f}")
    print(f"• Capital final: ${final_value:,.2f}")
    print(f"• P&L: ${pnl:,.2f} ({roi:.2f}%)")
    print(f"• Ratio Sharpe: {sharpe.get('sharperatio', 0):.2f}")
    
    # Préparer le rapport
    report = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'pnl': pnl,
        'roi': roi,
        'sharpe_ratio': sharpe['sharperatio'] if sharpe else 0,
        'max_drawdown': drawdown['max']['drawdown'] if drawdown else 0,
        'trades': {
            'total': trades.total.total if hasattr(trades, 'total') else 0,
            'won': trades.won.total if hasattr(trades, 'won') else 0,
            'lost': trades.lost.total if hasattr(trades, 'lost') else 0,
            'win_rate': trades.won.total / trades.total.total if hasattr(trades, 'total') and trades.total.total > 0 else 0
        }
    }
    
    # Afficher les résultats détaillés
    print("\n📊 Résultats détaillés du backtest:")
    print("="*50)
    print(f"• Capital final: ${final_value:,.2f}")
    print(f"• P&L: ${pnl:,.2f}")
    print(f"• ROI: {roi:.2f}%")
    print(f"• Ratio Sharpe: {report['sharpe_ratio']:.2f}")
    print(f"• Drawdown max: {report['max_drawdown']:.2f}%")
    print("\n📈 Analyse des trades:")
    print(f"• Total trades: {report['trades']['total']}")
    print(f"• Trades gagnants: {report['trades']['won']}")
    print(f"• Trades perdants: {report['trades']['lost']}")
    print(f"• Win rate: {report['trades']['win_rate']:.2%}")
    
    # Vérifier si le drawdown bloque les trades
    if report['trades']['total'] == 0:
        print("\n⚠️ Attention: Aucun trade n'a été exécuté!")
        print("Causes possibles:")
        print("• Seuil de confiance trop élevé")
        print("• Conditions de gestion du risque trop strictes")
        print("• Problème avec les signaux du modèle")
    elif report['max_drawdown'] == 0:
        print("\n⚠️ Attention: Drawdown maximum à 0%")
        print("Causes possibles:")
        print("• Trades bloqués par la gestion du risque")
        print("• Problème dans le calcul du drawdown")
    
    return report

def _prepare_backtrader_data(self, data):
    """Convertit les données en format Backtrader"""
    # Convertir la colonne Date en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])
    
    # S'assurer que les prix sont au bon format
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        # Si les prix sont trop bas, les multiplier par 1000
        if data[col].mean() < 1000:
            data[col] = data[col] * 1000
        # Convertir en float64 pour éviter les problèmes de type
        data[col] = data[col].astype('float64')
    
    # Créer le feed de données
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime='Date',
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    
    # Debug des prix
    print("\n🔍 Vérification des prix:")
    print(f"Prix moyen: {data['Close'].mean():.4f}")
    print(f"Prix min: {data['Close'].min():.4f}")
    print(f"Prix max: {data['Close'].max():.4f}")
    
    return data_feed