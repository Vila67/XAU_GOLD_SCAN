import numpy as np
import pandas as pd
from src.preprocessing.data_processor import DataProcessor
from src.data_collection.historical_data import HistoricalDataLoader
from typing import Dict

def print_dataset_info(X: Dict[str, np.ndarray], y: np.ndarray, name: str = ""):
    """Affiche les informations sur un dataset"""
    print(f"\n{name} Dataset:")
    
    print("\nDonnées d'entrée (X):")
    for tf, data in X.items():
        print(f"\n{tf}:")
        print(f"- Shape: {data.shape}")
        print(f"- Type: {data.dtype}")
        print(f"- Range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"- Mean: {data.mean():.4f}")
        print(f"- Std: {data.std():.4f}")
    
    print(f"\nLabels (y):")
    print(f"- Shape: {y.shape}")
    print(f"- Classes: {np.unique(y)}")
    print(f"- Distribution:")
    for label in np.unique(y):
        count = np.sum(y == label)
        pct = count / len(y) * 100
        print(f"  * {label}: {count} ({pct:.2f}%)")

def test_data_processing():
    """Test complet du pipeline de préparation des données"""
    # Charger les données de test
    data_dict = load_test_data()
    
    # Créer le processeur
    processor = DataProcessor()
    
    # Paramètres de test
    sequence_length = 15  # Réduit pour le test
    prediction_horizon = 12
    train_size = 0.8
    
    # Préparer les données
    X_train, y_train, X_test, y_test = processor.prepare_training_data(
        data_dict,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        train_size=train_size
    )
    
    # Vérifier les distributions et tailles
    print("\nTraining Dataset:\n")
    print_dataset_info(X_train, y_train)
    
    print("\nTest Dataset:\n")
    print_dataset_info(X_test, y_test)
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = test_data_processing()
        print("\nTest terminé avec succès!")
        
    except Exception as e:
        print(f"\nErreur pendant le test: {str(e)}")
        raise 