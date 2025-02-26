from src.preprocessing.ml_preprocessor import MLPreprocessor
from src.models.price_predictor import GoldPricePredictor
from src.data_collection.historical_data import HistoricalDataCollector
from src.preprocessing.data_processor import GoldDataProcessor
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

def test_price_prediction():
    # Charger et préparer les données
    print("Chargement des données...")
    collector = HistoricalDataCollector()
    data = collector.load_kaggle_data('1D')
    
    print("\nPréparation des indicateurs techniques...")
    processor = GoldDataProcessor()
    data = processor.add_technical_indicators(data)
    
    # Préparer les données pour le ML
    print("\nPréparation des features pour le ML...")
    ml_preprocessor = MLPreprocessor()
    X = ml_preprocessor.prepare_features(data)
    y = ml_preprocessor.prepare_target(data)
    
    print(f"Shape des features: {X.shape}")
    print(f"Shape de la cible: {y.shape}")
    
    # Créer les séquences
    print("\nCréation des séquences temporelles...")
    sequence_length = 10
    n_features = len(X.columns)
    X_seq, y_seq = ml_preprocessor.create_sequences(X, y, sequence_length)
    
    print(f"Shape des séquences: X={X_seq.shape}, y={y_seq.shape}")
    
    # Split train/test
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    print(f"\nDonnées d'entraînement: {X_train.shape}")
    print(f"Données de test: {X_test.shape}")
    
    # Entraîner le modèle
    print("\nEntraînement du modèle...")
    predictor = GoldPricePredictor(sequence_length=sequence_length, n_features=n_features)
    history = predictor.train(X_train, y_train, epochs=50)
    
    # Faire des prédictions
    print("\nPrédiction sur l'ensemble de test...")
    y_pred = predictor.predict(X_test)
    
    # Évaluer le modèle
    metrics = predictor.evaluate(y_test, y_pred)
    print("\nMétriques d'évaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 6))
    plt.plot(y_test, label='Réel', alpha=0.5)
    plt.plot(y_pred, label='Prédit', linewidth=2)
    plt.title('Prédiction de la variation du prix de l\'or')
    plt.xlabel('Temps')
    plt.ylabel('Variation du prix (%)')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarder le graphique
    plt.savefig('data/processed/predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_price_prediction() 