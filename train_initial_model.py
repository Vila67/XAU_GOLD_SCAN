import numpy as np
import pandas as pd
from src.preprocessing.ml_preprocessor import MLPreprocessor
from src.models.ml_model import MLModel
from src.data_collection.historical_data import HistoricalDataLoader
import os
import logging
from datetime import datetime
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
import argparse
from imblearn.over_sampling import SMOTE
from keras.optimizers import Adam
import sys
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.backend import clear_session
import itertools
import random
from tensorflow.keras.regularizers import l1_l2
import pickle
import traceback
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from tensorflow.keras import backend as K
import gc
import tensorflow as tf
import time  # Ajout de l'import time pour mesurer la durée d'exécution
import json
from pathlib import Path

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configuration du logging global
def setup_logging():
    logger = logging.getLogger('TrainInitialModel')
    logger.setLevel(logging.DEBUG)

    # Handler pour fichier
    os.makedirs('logs/training', exist_ok=True)
    fh = logging.FileHandler(
        f'logs/training/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)

    # Handler pour console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class ClassAccuracyMonitor(Callback):
    def __init__(self, val_inputs, val_outputs, timeframes, class_weights):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_outputs = val_outputs
        self.timeframes = timeframes
        self.class_weights = class_weights
        self.best_weights = None
        self.best_balanced_acc = 0
    
    def on_epoch_end(self, epoch, logs=None):
        val_preds = self.model.predict(self.val_inputs)
        print(f"\nEpoch {epoch} - Analyse par timeframe:")
        
        total_balanced_acc = 0
        for i, tf in enumerate(self.timeframes):
            pred_classes = np.argmax(val_preds[i], axis=1)
            true_classes = np.argmax(self.val_outputs[i], axis=1)
            
            # Calcul des métriques par classe
            class_acc = np.zeros(3)  # Initialiser pour les 3 classes
            class_dist = np.zeros(3)  # Initialiser pour les 3 classes
            
            # Distribution des prédictions
            unique, counts = np.unique(pred_classes, return_counts=True)
            for cls, count in zip(unique, counts):
                class_dist[cls] = count / len(pred_classes)
            
            # Accuracy par classe
            for c in range(3):
                mask = (true_classes == c)
                if np.sum(mask) > 0:
                    class_acc[c] = np.mean(pred_classes[mask] == c)
                    print(f"{tf} - Classe {c}: Acc={class_acc[c]:.3f}, Dist={class_dist[c]:.3f}")
                else:
                    print(f"{tf} - Classe {c}: Pas d'échantillons")
            
            # Calcul de l'accuracy balancée (moyenne des accuracies non-nulles)
            valid_acc = class_acc[class_acc > 0]
            balanced_acc = np.mean(valid_acc) if len(valid_acc) > 0 else 0
            total_balanced_acc += balanced_acc
            
            # Ajustement des poids si nécessaire
            if tf == '5m':  # Focus sur 5m
                neutral_bias = class_dist[1] - 0.33  # Biais vers la classe neutre
                if neutral_bias > 0.1:  # Si trop de prédictions neutres
                    self.class_weights[1] *= 0.9  # Réduire le poids de la classe neutre
                    print(f"\n⚠️ Biais neutre détecté ({neutral_bias:.2f})")
                    print(f"Poids ajustés: {self.class_weights}")
                
                # Vérifier aussi les classes extrêmes
                if class_dist[0] < 0.2:  # Si pas assez de prédictions baissières
                    self.class_weights[0] *= 1.1
                    print(f"⚠️ Renforcement classe baissière ({class_dist[0]:.2f})")
                if class_dist[2] < 0.2:  # Si pas assez de prédictions haussières
                    self.class_weights[2] *= 1.1
                    print(f"⚠️ Renforcement classe haussière ({class_dist[2]:.2f})")
        
        # Sauvegarder les meilleurs poids
        avg_balanced_acc = total_balanced_acc / len(self.timeframes)
        if avg_balanced_acc > self.best_balanced_acc:
            self.best_balanced_acc = avg_balanced_acc
            self.best_weights = self.model.get_weights()
            print(f"\n✨ Nouveau meilleur modèle: {avg_balanced_acc:.4f}")

class WalkForwardSplit:
    def __init__(self, n_splits=5, train_size=0.8, gap=0):
        """
        n_splits: nombre de folds
        train_size: proportion de données pour l'entraînement
        gap: nombre d'échantillons à sauter entre train et validation
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.gap = gap
    
    def split(self, X, y=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Taille de chaque fold
        fold_size = n_samples // self.n_splits
        train_size = int(fold_size * self.train_size)
        
        for i in range(self.n_splits - 1):
            # Indices pour ce fold
            start_idx = i * fold_size
            train_end = start_idx + train_size
            val_start = train_end + self.gap
            val_end = start_idx + fold_size
            
            # Retourner les indices train/val
            yield (
                indices[start_idx:train_end],
                indices[val_start:val_end]
            )

class HyperparameterOptimizer:
    def __init__(self):
        self.param_ranges = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': [16, 32, 64],
            'lstm1_units': (32, 256),
            'lstm2_units': (16, 128),
            'dropout_rate': (0.1, 0.5),
            'patience': (10, 50),
            'l1_reg': (1e-6, 1e-3),
            'l2_reg': (1e-6, 1e-3)
        }
        self.n_trials = 20
        self.n_splits = 5
        self.gap = 12  # Gap de 1 heure entre train et validation
        self.memory_growth = True  # Activer la croissance mémoire dynamique
    
    def cleanup_memory(self):
        """Nettoie la mémoire GPU/CPU"""
        K.clear_session()
        gc.collect()
    
    def objective(self, trial, train_inputs, train_outputs, timeframes):
        """Fonction objectif pour Optuna"""
        try:
            # Suggérer des valeurs pour les hyperparamètres
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 
                                                   *self.param_ranges['learning_rate'], 
                                                   log=True),
                'batch_size': trial.suggest_categorical('batch_size', 
                                                      self.param_ranges['batch_size']),
                'lstm1_units': trial.suggest_int('lstm1_units', 
                                               *self.param_ranges['lstm1_units']),
                'lstm2_units': trial.suggest_int('lstm2_units', 
                                               *self.param_ranges['lstm2_units']),
                'dropout_rate': trial.suggest_float('dropout_rate', 
                                                  *self.param_ranges['dropout_rate']),
                'patience': trial.suggest_int('patience', 
                                            *self.param_ranges['patience']),
                'l1_reg': trial.suggest_float('l1_reg', 
                                            *self.param_ranges['l1_reg'], 
                                            log=True),
                'l2_reg': trial.suggest_float('l2_reg', 
                                            *self.param_ranges['l2_reg'], 
                                            log=True)
            }
            
            print(f"\nEssai #{trial.number}")
            print("Paramètres:", params)
            
            # Validation croisée temporelle
            wf_split = WalkForwardSplit(
                n_splits=self.n_splits,
                train_size=0.8,
                gap=self.gap
            )
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(wf_split.split(train_inputs[0]), 1):
                try:
                    print(f"\nFold {fold}/{self.n_splits}")
                    self.cleanup_memory()  # Nettoyage avant chaque fold
                    
                    # Préparer les données pour ce fold
                    fold_train_inputs = [X[train_idx] for X in train_inputs]
                    fold_train_outputs = [y[train_idx] for y in train_outputs]
                    fold_val_inputs = [X[val_idx] for X in train_inputs]
                    fold_val_outputs = [y[val_idx] for y in train_outputs]
                    
                    # Créer et entraîner le modèle
                    model = self.create_model(params, timeframes)
                    
                    # Callbacks avec pruning Optuna
                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss',
                            patience=params['patience'],
                            restore_best_weights=True,
                            min_delta=0.0001
                        ),
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.2,
                            patience=params['patience']//2,
                            min_lr=1e-7
                        ),
                        TFKerasPruningCallback(trial, 'val_loss')
                    ]
                    
                    # Entraînement avec gestion de la mémoire
                    try:
                        history = model.fit(
                            fold_train_inputs,
                            fold_train_outputs,
                            validation_data=(fold_val_inputs, fold_val_outputs),
                            batch_size=params['batch_size'],
                            epochs=50,
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        val_loss = min(history.history['val_loss'])
                        fold_scores.append(val_loss)
                        print(f"• Validation loss: {val_loss:.4f}")
                        
                        # Pruning précoce si les performances sont mauvaises
                        trial.report(val_loss, fold)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        
                    finally:
                        # Nettoyage explicite du modèle
                        del model
                        del history
                        self.cleanup_memory()
                    
                except optuna.TrialPruned:
                    print("❌ Essai arrêté prématurément (pruning)")
                    raise
                except Exception as e:
                    print(f"❌ Échec du fold {fold}: {str(e)}")
                    fold_scores.append(float('inf'))
                finally:
                    # Nettoyage final du fold
                    self.cleanup_memory()
            
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            print(f"\nScore moyen: {mean_score:.4f} (±{std_score:.4f})")
            
            return mean_score
            
        except Exception as e:
            print(f"❌ Erreur dans l'objectif: {str(e)}")
            raise
        finally:
            # Nettoyage final de l'essai
            self.cleanup_memory()
    
    def optimize(self, train_inputs, train_outputs, timeframes):
        """Optimise les hyperparamètres avec Optuna"""
        try:
            # Configuration de la mémoire GPU si disponible
            if self.memory_growth:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("✓ Croissance mémoire GPU activée")
            
            print(f"\nOptimisation des hyperparamètres avec Optuna:")
            print(f"• {self.n_trials} essais maximum")
            print(f"• {self.n_splits} folds temporels")
            print(f"• Gap de {self.gap} échantillons entre train et validation")
            
            # Créer une étude Optuna
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=3
                )
            )
            
            # Lancer l'optimisation
            study.optimize(
                lambda trial: self.objective(trial, train_inputs, train_outputs, timeframes),
                n_trials=self.n_trials,
                timeout=None,  # Pas de limite de temps
                catch=(Exception,)
            )
            
            # Analyse des résultats
            print("\nRésultats de l'optimisation:")
            
            print("\nMeilleurs paramètres trouvés:")
            for param, value in study.best_params.items():
                print(f"• {param}: {value}")
            
            print(f"\nMeilleur score: {study.best_value:.4f}")
            
            # Statistiques des essais
            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
            
            print("\nStatistiques des essais:")
            print(f"• Complétés: {len(completed_trials)}")
            print(f"• Arrêtés (pruning): {len(pruned_trials)}")
            
            # Importance des paramètres
            importance = optuna.importance.get_param_importances(study)
            print("\nImportance des paramètres:")
            for param, score in importance.items():
                print(f"• {param}: {score:.3f}")
            
            return study.best_params
            
        finally:
            # Nettoyage final
            self.cleanup_memory()

def load_historical_data():
    """Charge les données historiques depuis les fichiers"""
    loader = HistoricalDataLoader()
    
    logger = logging.getLogger('TrainInitialModel')
    logger.info("\nChargement des données historiques...")
    
    try:
        # Utiliser directement les timeframes définis dans HistoricalDataLoader
        data_dict = loader.load_multi_timeframe_data()
        
        # Vérification et logging des données chargées
        logger.info("\nRésumé des données chargées:")
        for tf in loader.timeframes:  # Utiliser les timeframes de l'instance
            if tf not in data_dict or data_dict[tf].empty:
                logger.warning(f"⚠️ Pas de données pour {tf}")
            else:
                df = data_dict[tf]
                logger.info(f"\n✓ Données {tf}:")
                logger.info(f"• Lignes: {len(df):,}")
                logger.info(f"• Période: {df.index.min()} à {df.index.max()}")
                logger.info(f"• Colonnes: {', '.join(df.columns)}")
        
        if not data_dict:
            raise ValueError("Aucune donnée historique n'a pu être chargée")
        
        logger.info("\n✓ Chargement des données terminé avec succès")
        return data_dict
    
    except Exception as e:
        logger.error(f"\n❌ Erreur lors du chargement des données:")
        logger.error(f"• Type: {type(e).__name__}")
        logger.error(f"• Message: {str(e)}")
        raise

def save_checkpoint(data, name, timestamp):
    """Sauvegarde un point de contrôle"""
    checkpoint_dir = f"checkpoints/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        checkpoint_path = f"{checkpoint_dir}/{name}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Checkpoint sauvegardé: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du checkpoint {name}: {str(e)}")

def load_checkpoint(name, timestamp):
    """Charge un point de contrôle"""
    checkpoint_path = f"checkpoints/{timestamp}/{name}.pkl"
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

class CheckpointCallback(Callback):
    def __init__(self, timestamp, save_freq=5):
        super().__init__()
        self.timestamp = timestamp
        self.save_freq = save_freq
        
        # Créer le répertoire de base pour les checkpoints
        self.checkpoint_dir = f"checkpoints/{self.timestamp}/model"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # Ajouter l'extension .keras au nom du fichier
            model_path = os.path.join(
                self.checkpoint_dir, 
                f"epoch_{epoch+1}.keras"  # Utiliser .keras au lieu de .h5
            )
            self.model.save(model_path)
            print(f"\n✓ Modèle sauvegardé à l'epoch {epoch+1}")

def setup_kaggle_credentials():
    """Configure les credentials Kaggle"""
    try:
        # Chemin vers le fichier kaggle.json
        kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not os.environ.get('KAGGLE_API_KEY') and kaggle_path.exists():
            with open(kaggle_path) as f:
                credentials = json.load(f)
                os.environ['KAGGLE_USERNAME'] = credentials['username']
                os.environ['KAGGLE_API_KEY'] = credentials['key']
            print("✓ Credentials Kaggle chargés avec succès")
        elif not kaggle_path.exists():
            raise FileNotFoundError(f"Fichier kaggle.json non trouvé dans {kaggle_path}")
            
    except Exception as e:
        raise Exception(f"Erreur lors du chargement des credentials Kaggle: {str(e)}")

def train_initial_model(debug_mode=False):
    logger = setup_logging()
    
    # Générer un timestamp unique pour cette session d'entraînement
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info("Début de l'entraînement du modèle initial")
        preprocessor = MLPreprocessor(instrument='XAUUSD')
        
        # Chargement des données avec point de sauvegarde
        try:
            data_dict = load_checkpoint('data_dict', timestamp)
            if data_dict is None:
                data_dict = load_historical_data()
                save_checkpoint(data_dict, 'data_dict', timestamp)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise
        
        # Création des séquences avec point de sauvegarde
        try:
            sequences = load_checkpoint('sequences', timestamp)
            if sequences is None:
                sequences = preprocessor.create_sequences(
                    data_dict,
                    sequence_length=15,
                    prediction_horizon=12
                )
                save_checkpoint(sequences, 'sequences', timestamp)
        except Exception as e:
            logger.error(f"Erreur lors de la création des séquences: {str(e)}")
            raise
        
        # Optimisation des hyperparamètres avec point de sauvegarde
        try:
            best_params = load_checkpoint('best_params', timestamp)
            if best_params is None:
                best_params = preprocessor.optimize_hyperparameters(
                    train_data=sequences['train'],
                    n_combinations=20 if not debug_mode else 2
                )
                save_checkpoint(best_params, 'best_params', timestamp)
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation des hyperparamètres: {str(e)}")
            raise
        
        # Création et entraînement du modèle
        try:
            reference_tf = list(sequences['train']['X'].keys())[0]
            input_shape = (
                sequences['train']['X'][reference_tf].shape[1],
                sequences['train']['X'][reference_tf].shape[2]
            )
            
            learning_rate = best_params['learning_rate']
            model = MLModel(
                input_shape=input_shape,
                learning_rate=learning_rate,
                batch_size=32
            )
            
            # Préparer les données
            train_X = preprocessor.prepare_data(sequences['train']['X'])
            test_X = preprocessor.prepare_data(sequences['test']['X'])
            
            # Configuration des callbacks
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
                ),
                ModelCheckpoint(
                    filepath=f'models/best_model_{timestamp}.keras',  # Ajouter timestamp et .keras
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                ),
                CheckpointCallback(timestamp)
            ]
            
            # Entraînement avec les callbacks mis à jour
            history = model.train_multi_timeframe(
                train_X,
                sequences['train']['y'],
                validation_data=(test_X, sequences['test']['y']),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Sauvegarder le modèle final
            final_model_path = f'models/final_model_{timestamp}.keras'
            model.save(final_model_path)
            logger.info(f"✓ Modèle final sauvegardé: {final_model_path}")
            
            # Sauvegarder l'historique d'entraînement
            save_checkpoint(history.history, 'training_history', timestamp)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
            raise
        
        # Backtest avec point de sauvegarde
        try:
            backtest_results = backtest_model(
                model,
                sequences['test']['X'],
                sequences['test']['y'],
                data_dict[reference_tf]
            )
            save_checkpoint(backtest_results, 'backtest_results', timestamp)
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {str(e)}")
            raise
        
        return model, history, backtest_results
        
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        # Tenter de sauvegarder l'état actuel en cas d'erreur
        try:
            error_state = {
                'error': str(e),
                'timestamp': timestamp,
                'traceback': traceback.format_exc()
            }
            save_checkpoint(error_state, 'error_state', timestamp)
        except:
            pass
        raise

def adjust_threshold_for_timeframe(base_threshold, timeframe):
    multipliers = {'5m': 1.0, '15m': 1.5, '30m': 2.0, '1h': 3.0, 
                  '4h': 5.0, '1d': 10.0, '1w': 20.0, '1M': 40.0}
    return base_threshold * multipliers.get(timeframe, 1.0)

def calculate_dynamic_costs(price_data, lookback=14):
    """Calcule les coûts dynamiques basés sur l'ATR"""
    # Calcul de l'ATR
    high = price_data['High']
    low = price_data['Low']
    close = price_data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    atr = tr.rolling(window=lookback).mean()
    
    # Calcul des coûts dynamiques
    spread_pips = (atr * 0.1).clip(1.5, 5.0)  # Entre 1.5 et 5.0 pips
    slippage_pips = (atr * 0.05).clip(0.5, 2.0)  # Entre 0.5 et 2.0 pips
    
    return {
        'spread_pips': spread_pips,
        'slippage_pips': slippage_pips,
        'atr': atr
    }

def calculate_realistic_pnl(entry_price, exit_price, position_size, direction, 
                          spread_pips, slippage_pips, volatility):
    """Calcule le PNL avec des coûts dynamiques"""
    # Configuration des coûts
    pip_value = 0.01  # 1 pip = 0.01 pour XAU/USD
    pip_dollar_value = 1.0  # 1 pip = 1$ pour 0.1 lot
    commission_per_lot = 6.0  # Commission fixe par lot
    
    # Conversion des pips en points
    spread = spread_pips * pip_value
    slippage = slippage_pips * pip_value * direction
    
    # Calcul des coûts de transaction
    commission = commission_per_lot * position_size
    
    print("\nCalcul PNL détaillé (XAU/USD):")
    print(f"• Prix d'entrée brut: {entry_price:.2f}")
    print(f"• Prix de sortie brut: {exit_price:.2f}")
    print(f"• Direction: {'LONG' if direction > 0 else 'SHORT'}")
    print(f"• Taille position: {position_size:.2f} lots")
    print(f"• Commission: ${commission:.2f}")
    print(f"• Spread: {spread_pips:.1f} pips = ${spread_pips * pip_dollar_value:.2f}")
    print(f"• Slippage: {slippage_pips:.1f} pips = ${slippage_pips * position_size * 100:.2f}")
    print(f"• Volatilité (ATR): {volatility:.1f} pips")
    
    # Application des coûts
    if direction == 1:  # Long
        real_entry = entry_price + spread + slippage
        real_exit = exit_price - spread - slippage
    else:  # Short
        real_entry = entry_price - spread - slippage
        real_exit = exit_price + spread + slippage
    
    # Calcul du PNL
    price_diff = (real_exit - real_entry) * direction
    price_diff_pips = price_diff / pip_value
    
    # Calcul du PNL final avec commission
    pnl = price_diff_pips * pip_dollar_value * position_size * 100 - commission * 2
    
    print(f"• Prix d'entrée ajusté: {real_entry:.2f}")
    print(f"• Prix de sortie ajusté: {real_exit:.2f}")
    print(f"• Différence: {price_diff:.2f} points = {price_diff_pips:.1f} pips")
    print(f"• PNL net: ${pnl:.2f}")
    
    return pnl

def calculate_position_size(risk_amount, stop_loss_pips, current_price, volatility):
    """Calcule la taille de position avec ajustement volatilité"""
    # Pour XAU/USD:
    pip_dollar_value = 1.0  # 1 pip = 1$ pour 0.1 lot
    
    # Ajustement du risque selon la volatilité
    volatility_factor = min(max(1.0, volatility/20), 2.0)  # Entre 1.0 et 2.0
    adjusted_risk = risk_amount / volatility_factor
    
    # Calcul de la taille de position
    position_size = adjusted_risk / (stop_loss_pips * pip_dollar_value * 100)
    
    print(f"\nCalcul taille position (XAU/USD):")
    print(f"• Montant risqué: ${risk_amount:.2f}")
    print(f"• Montant ajusté (volatilité): ${adjusted_risk:.2f}")
    print(f"• Stop loss: {stop_loss_pips} pips")
    print(f"• Prix actuel: {current_price:.2f}")
    print(f"• Valeur pip: ${pip_dollar_value:.2f} pour 0.1 lot")
    print(f"• Volatilité (ATR): {volatility:.1f} pips")
    print(f"• Taille calculée: {position_size:.4f} lots")
    
    # Limiter la taille maximale et arrondir à 0.01 lot
    final_size = min(round(position_size, 2), 1.0)
    if final_size != position_size:
        print(f"• Taille ajustée: {final_size:.2f} lots (max 1.0)")
    
    return final_size

class BacktestManager:
    def __init__(self, initial_balance=100000, risk_per_trade=0.02):
        # Configuration initiale
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pips = 20
        self.take_profit_pips = 40
        self.max_positions = 3
        self.min_position_size = 0.01
        self.max_position_size = 1.0
        self.max_loss_pct = 0.5
        self.max_drawdown_pct = 0.3
        
        # Paramètres de filtrage
        self.confidence_threshold_long = 0.5
        self.confidence_threshold_short = 0.5
        self.neutral_zone = 0.4
        self.min_bars_between_trades = 12
        self.max_trades_per_day = 5
        self.cooldown_after_loss = 24
        self.daily_loss_limit = -0.02
        
        # Variables de suivi
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.open_positions = []
        self.trades = []
        self.equity_curve = [initial_balance]
        self.daily_returns = []
        
        # Variables journalières
        self.current_day = None
        self.day_start_equity = initial_balance
        self.trades_today = 0
        self.daily_pnl = 0
        self.last_trade_bar = -self.min_bars_between_trades
        self.last_loss_bar = -self.cooldown_after_loss
    
    def can_open_position(self, current_bar):
        """Vérifie si on peut ouvrir une nouvelle position"""
        conditions = {
            "max_positions": len(self.open_positions) < self.max_positions,
            "min_bars": current_bar - self.last_trade_bar >= self.min_bars_between_trades,
            "cooldown": current_bar - self.last_loss_bar >= self.cooldown_after_loss,
            "daily_trades": self.trades_today < self.max_trades_per_day,
            "daily_loss": self.daily_pnl/self.initial_balance > self.daily_loss_limit
        }
        
        can_trade = all(conditions.values())
        
        if not can_trade:
            reasons = [k for k, v in conditions.items() if not v]
            print(f"⛔ Pas de nouveau trade: {', '.join(reasons)}")
        
        return can_trade
    
    def analyze_signal(self, pred_probs):
        """Analyse les probabilités pour déterminer la direction"""
        down_prob, neutral_prob, up_prob = pred_probs
        
        print(f"\nAnalyse des probabilités:")
        print(f"• Baisse: {down_prob:.1%}")
        print(f"• Neutre: {neutral_prob:.1%}")
        print(f"• Hausse: {up_prob:.1%}")
        
        if neutral_prob >= self.neutral_zone:
            print("Signal trop neutre")
            return 0
            
        if up_prob > self.confidence_threshold_long and up_prob > down_prob * 1.5:
            print("Signal LONG détecté")
            return 1
        elif down_prob > self.confidence_threshold_short and down_prob > up_prob * 1.5:
            print("Signal SHORT détecté")
            return -1
            
        return 0
    
    def open_position(self, current_price, direction, current_time, pred_probs, costs):
        """Ouvre une nouvelle position"""
        risk_amount = max(self.current_balance * self.risk_per_trade, 0)
        
        position_size = calculate_position_size(
            risk_amount,
            self.stop_loss_pips,
            current_price,
            costs['atr']
        )
        
        if position_size < self.min_position_size:
            print(f"⚠️ Taille position trop petite: {position_size:.4f} lots")
            return False
        
        position = {
            'entry_price': current_price,
            'direction': direction,
            'size': position_size,
            'entry_time': current_time,
            'pred_probs': pred_probs.copy(),
            'spread': costs['spread_pips'],
            'slippage': costs['slippage_pips'],
            'volatility': costs['atr']
        }
        
        self.open_positions.append(position)
        self.trades_today += 1
        
        print(f"\n🔔 Nouveau trade ({current_time}):")
        print(f"• Direction: {'LONG' if direction > 0 else 'SHORT'}")
        print(f"• Prix: {current_price:.2f}")
        print(f"• Taille: {position_size:.2f}")
        print(f"• Trade #{self.trades_today} du jour")
        print(f"• PnL journalier: {self.daily_pnl:,.2f} ({self.daily_pnl/self.initial_balance:.1%})")
        
        return True
    
    def close_position(self, position, exit_price, exit_type, current_bar):
        """Ferme une position existante"""
        pnl = calculate_realistic_pnl(
            position['entry_price'],
            exit_price,
            position['size'],
            position['direction'],
            position['spread'],
            position['slippage'],
            position['volatility']
        )
        
        self.current_balance += pnl
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.last_loss_bar = current_bar
        
        profit_pips = self.take_profit_pips if exit_type == 'take_profit' else -self.stop_loss_pips
        if position['direction'] == -1:  # Inverser pour les shorts
            profit_pips = -profit_pips
        
        trade = {
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'direction': position['direction'],
            'exit_type': exit_type,
            'profit_pips': profit_pips,
            'entry_time': position['entry_time'],
            'exit_time': current_bar
        }
        
        self.trades.append(trade)
        self.open_positions.remove(position)
        
        print(f"\n🔔 Trade fermé ({'LONG' if position['direction'] == 1 else 'SHORT'}):")
        print(f"• Raison: {'✅ Take Profit' if exit_type == 'take_profit' else '❌ Stop Loss'}")
        print(f"• Prix entrée: {position['entry_price']:.2f}")
        print(f"• Prix sortie: {exit_price:.2f}")
        print(f"• {'Gain' if pnl > 0 else 'Perte'}: {abs(profit_pips):.1f} pips")
        print(f"• PNL: {pnl:,.2f} USD")
    
    def update_daily_stats(self, current_time):
        """Met à jour les statistiques journalières"""
        if self.current_day != current_time.date():
            if self.current_day is not None:
                daily_return = (self.current_balance - self.day_start_equity) / self.day_start_equity
                self.daily_returns.append(daily_return)
                print(f"\nFin de journée {self.current_day}:")
                print(f"• Return: {daily_return:.2%}")
                print(f"• PnL: ${self.current_balance - self.day_start_equity:,.2f}")
            
            self.current_day = current_time.date()
            self.day_start_equity = self.current_balance
            self.trades_today = 0
            self.daily_pnl = 0
            print(f"\nNouveau jour: {self.current_day}")
            print(f"• Balance: ${self.current_balance:,.2f}")
    
    def check_stop_conditions(self):
        """Vérifie les conditions d'arrêt du backtest"""
        if self.current_balance <= 0:
            print(f"\n❌ RUINE: Balance négative ({self.current_balance:,.2f} USD)")
            return True
        
        loss_pct = (self.initial_balance - self.current_balance) / self.initial_balance
        if loss_pct >= self.max_loss_pct:
            print(f"\n⛔ ARRÊT: Perte maximale atteinte ({loss_pct:.1%})")
            return True
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown >= self.max_drawdown_pct:
            print(f"\n⛔ ARRÊT: Drawdown maximum atteint ({current_drawdown:.1%})")
            return True
        
        return False

def backtest_model(model, X_test_list, y_test, original_data_5m, confidence_threshold=0.3):
    """Fonction principale de backtest"""
    try:
        # Initialisation
        bt = BacktestManager()
        costs = calculate_dynamic_costs(original_data_5m)
        
        # Prédictions du modèle
        predictions = model.predict(X_test_list)
        
        # Si predictions est un seul array, le mettre dans une liste
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Préparation des données temporelles
        val_start_idx = len(original_data_5m) - len(y_test)
        dates = original_data_5m.index[val_start_idx:val_start_idx + len(predictions[0])]
        close_prices = original_data_5m['Close'].values[val_start_idx:val_start_idx + len(predictions[0])]
        
        print("\nDébut du backtest...")
        print(f"Période: {dates[0]} à {dates[-1]}")
        
        # Boucle principale
        for i in range(len(predictions[0])):
            current_price = float(close_prices[i])
            current_time = dates[i]
            
            # Mise à jour des statistiques journalières
            bt.update_daily_stats(current_time)
            
            # Gestion des positions existantes
            for pos in bt.open_positions[:]:  # Copie de la liste pour éviter les problèmes de modification
                total_cost = (pos['spread'] + pos['slippage']) * 0.01  # pip_value = 0.01
                sl_price = pos['entry_price'] - (bt.stop_loss_pips * 0.01 + total_cost) * pos['direction']
                tp_price = pos['entry_price'] + (bt.take_profit_pips * 0.01 - total_cost) * pos['direction']
                
                # Vérification des niveaux
                if pos['direction'] == 1:  # Long
                    if current_price <= sl_price:
                        bt.close_position(pos, sl_price, 'stop_loss', i)
                    elif current_price >= tp_price:
                        bt.close_position(pos, tp_price, 'take_profit', i)
                else:  # Short
                    if current_price >= sl_price:
                        bt.close_position(pos, sl_price, 'stop_loss', i)
                    elif current_price <= tp_price:
                        bt.close_position(pos, tp_price, 'take_profit', i)
            
            # Mise à jour de l'equity curve
            bt.peak_balance = max(bt.peak_balance, bt.current_balance)
            bt.equity_curve.append(bt.current_balance)
            
            # Vérification des conditions d'arrêt
            if bt.check_stop_conditions():
                break
            
            # Analyse du signal
            if bt.can_open_position(i):
                # Obtenir la prédiction pour ce point temporel
                current_pred = predictions[0][i]
                
                # Déterminer la direction en fonction de la prédiction
                if isinstance(current_pred, (np.ndarray, list)):
                    # Si la prédiction est un vecteur de probabilités
                    pred_class = np.argmax(current_pred)
                    confidence = current_pred[pred_class]
                    
                    if confidence >= confidence_threshold:
                        direction = 1 if pred_class == 2 else (-1 if pred_class == 0 else 0)
                else:
                    # Si la prédiction est une seule valeur
                    direction = 1 if current_pred > confidence_threshold else (-1 if current_pred < -confidence_threshold else 0)
                
                if direction != 0:
                    bt.open_position(
                        current_price,
                        direction,
                        current_time,
                        current_pred,
                        {
                            'atr': costs['atr'].iloc[i],
                            'spread_pips': costs['spread_pips'].iloc[i],
                            'slippage_pips': costs['slippage_pips'].iloc[i]
                        }
                    )
        
        # Calcul des statistiques finales
        return calculate_backtest_statistics(bt)
        
    except Exception as e:
        print(f"\n❌ Erreur dans backtest_model: {str(e)}")
        raise

def calculate_backtest_statistics(bt):
    """Calcule les statistiques finales du backtest"""
    try:
        # Calcul des métriques de base
        final_balance = bt.current_balance
        total_pnl = final_balance - bt.initial_balance
        roi = (total_pnl / bt.initial_balance) * 100
        
        # Calcul du drawdown maximum
        equity_curve = np.array(bt.equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Calcul des statistiques des trades
        n_trades = len(bt.trades)
        if n_trades > 0:
            winning_trades = sum(1 for pos in bt.trades if pos['pnl'] > 0)
            losing_trades = sum(1 for pos in bt.trades if pos['pnl'] <= 0)
            win_rate = (winning_trades / n_trades) * 100
            
            # Calcul du profit factor
            gross_profit = sum(pos['pnl'] for pos in bt.trades if pos['pnl'] > 0)
            gross_loss = abs(sum(pos['pnl'] for pos in bt.trades if pos['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calcul de l'average trade
            avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
            
        else:
            winning_trades = losing_trades = win_rate = 0
            profit_factor = avg_win = avg_loss = 0
        
        # Calcul des rendements journaliers
        daily_returns = np.array(bt.daily_returns)
        if len(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
            max_daily_return = np.max(daily_returns) * 100
            min_daily_return = np.min(daily_returns) * 100
        else:
            sharpe_ratio = max_daily_return = min_daily_return = 0
        
        # Création du rapport
        stats = {
            'initial_balance': bt.initial_balance,
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': {
                'total': n_trades,
                'winning': winning_trades,
                'losing': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'daily': {
                'best': max_daily_return,
                'worst': min_daily_return,
                'total_days': len(daily_returns)
            }
        }
        
        # Affichage du rapport
        print("\n📊 Résultats du backtest:")
        print("="*50)
        print(f"Balance initiale: ${bt.initial_balance:,.2f}")
        print(f"Balance finale: ${final_balance:,.2f}")
        print(f"Profit/Perte: ${total_pnl:,.2f} ({roi:.2f}%)")
        print(f"Drawdown maximum: {max_drawdown:.2f}%")
        print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        
        print("\n📈 Statistiques des trades:")
        print(f"Nombre total de trades: {n_trades}")
        print(f"Trades gagnants: {winning_trades}")
        print(f"Trades perdants: {losing_trades}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Profit factor: {profit_factor:.2f}")
        print(f"Gain moyen: ${avg_win:,.2f}")
        print(f"Perte moyenne: ${avg_loss:,.2f}")
        
        print("\n📅 Statistiques journalières:")
        print(f"Meilleur jour: {max_daily_return:.2f}%")
        print(f"Pire jour: {min_daily_return:.2f}%")
        print(f"Nombre de jours de trading: {len(daily_returns)}")
        
        return stats
        
    except Exception as e:
        print(f"\n❌ Erreur dans calculate_backtest_statistics: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Configurer les credentials Kaggle avant tout
        setup_kaggle_credentials()
        
        # Créer le dossier logs s'il n'existe pas
        os.makedirs('logs', exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Lancer l'entraînement directement
        start_time = time.time()
        train_initial_model()
        
        # Calculer et afficher le temps total d'exécution
        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n⏱️ Temps total d'exécution: {:02}h {:02}min {:02.0f}s".format(
            int(hours), int(minutes), seconds))
        
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {str(e)}")
        sys.exit(1)