import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Conv1D, Bidirectional, 
                                   Attention, LayerNormalization, Add, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

class GoldPricePredictor:
    def __init__(self, sequence_length=10, n_features=29):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self):
        # Entrées pour chaque timeframe
        input_1d = Input(shape=(self.sequence_length, self.n_features), name='daily_input')
        input_4h = Input(shape=(self.sequence_length * 6, self.n_features), name='4h_input')
        input_1h = Input(shape=(self.sequence_length * 24, self.n_features), name='1h_input')
        
        # Fonction pour créer une branche de traitement
        def create_timeframe_branch(inputs, name):
            x = Conv1D(32, kernel_size=2, activation='relu', padding='same')(inputs)
            x = Conv1D(32, kernel_size=4, activation='relu', padding='same')(x)
            x = LayerNormalization()(x)
            
            # Attention locale
            lstm = Bidirectional(LSTM(32, return_sequences=True))(x)
            attention = Attention()([lstm, lstm])
            
            # Réduire la séquence à la même longueur que daily
            if name != 'daily':
                pool_size = 6 if name == '4h' else 24
                x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(attention)
            else:
                x = attention
                
            return x
        
        # Branches pour chaque timeframe
        daily_branch = create_timeframe_branch(input_1d, 'daily')
        h4_branch = create_timeframe_branch(input_4h, '4h')
        h1_branch = create_timeframe_branch(input_1h, '1h')
        
        # Fusion des branches
        merged = Concatenate()([daily_branch, h4_branch, h1_branch])
        
        # Traitement commun
        x = Conv1D(64, kernel_size=1, padding='same')(merged)
        x = LayerNormalization()(x)
        
        # BiLSTM avec attention globale
        lstm1 = Bidirectional(LSTM(64, return_sequences=True))(x)
        attention1 = Attention()([lstm1, lstm1])
        x = LayerNormalization()(attention1)
        x = Dropout(0.15)(x)
        
        # Second BiLSTM
        lstm2 = Bidirectional(LSTM(32))(x)
        x = LayerNormalization()(lstm2)
        x = Dropout(0.15)(x)
        
        # Dense layers avec skip connection
        dense1 = Dense(32, activation='relu')(x)
        x = LayerNormalization()(dense1)
        dense2 = Dense(16, activation='relu')(x)
        dense_skip = Dense(16)(x)
        x = Add()([dense2, dense_skip])
        x = LayerNormalization()(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[input_1d, input_4h, input_1h], outputs=outputs)
        
        def multi_timeframe_loss(y_true, y_pred):
            """Loss function adaptée aux différentes échelles temporelles"""
            # MSE de base
            mse = tf.square(y_pred - y_true)
            
            # Variations à différentes échelles
            def calculate_variation(tensor, scale):
                if tf.shape(tensor)[0] > scale:
                    variation = tf.abs(tensor[scale:] - tensor[:-scale])
                    padding = tf.zeros([scale, 1])
                    return tf.concat([variation, padding], axis=0)
                return tensor
            
            # Calculer les variations pour chaque timeframe
            variations = [
                calculate_variation(y_true, 1),  # Daily
                calculate_variation(y_true, 6),  # 4h
                calculate_variation(y_true, 24)  # 1h
            ]
            
            # Poids adaptatifs pour chaque timeframe
            weights = [
                tf.where(tf.abs(var) > 0.01, 1.5, 1.0)
                for var in variations
            ]
            
            # Combiner les poids
            total_weight = tf.reduce_prod(weights, axis=0)
            weighted_loss = mse * total_weight
            
            return tf.reduce_mean(weighted_loss)
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer,
                     loss=multi_timeframe_loss,
                     metrics=['mae', 'mse'])
        
        return model
    
    def create_lr_scheduler(self, epochs):
        initial_learning_rate = 0.0003
        decay_steps = epochs * 0.5
        decay_rate = 0.2
        
        def lr_scheduler(epoch):
            if epoch < decay_steps:
                return initial_learning_rate
            else:
                return initial_learning_rate * (decay_rate ** ((epoch - decay_steps) / 10))
        
        return tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    def train(self, X_train, y_train, epochs=150, batch_size=32, validation_split=0.2):
        """
        Adapte l'entrée unique en format multi-timeframes
        """
        print(f"Shape des données d'entraînement: X={X_train.shape}, y={y_train.shape}")
        
        # Convertir l'entrée unique en dictionnaire multi-timeframes
        X_train_dict = {
            'daily_input': X_train,
            '4h_input': np.repeat(X_train, 6, axis=1),  # Répéter les données pour simuler 4h
            '1h_input': np.repeat(X_train, 24, axis=1)  # Répéter les données pour simuler 1h
        }
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        lr_scheduler = self.create_lr_scheduler(epochs)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'data/models/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        history = self.model.fit(
            X_train_dict,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, lr_scheduler, checkpoint],
            verbose=1
        )
        
        self.plot_training_history(history)
        return history
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('data/processed/training_history.png')
        plt.close()
    
    def predict(self, X_test):
        """
        Adapte l'entrée de test au format multi-timeframes
        """
        print(f"Shape des données de test: X={X_test.shape}")
        
        # Convertir l'entrée de test en dictionnaire multi-timeframes
        X_test_dict = {
            'daily_input': X_test,
            '4h_input': np.repeat(X_test, 6, axis=1),
            '1h_input': np.repeat(X_test, 24, axis=1)
        }
        
        return self.model.predict(X_test_dict)
    
    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae
        } 