import os
import pandas as pd
import numpy as np
from src.data_collection.historical_data import HistoricalDataLoader
from src.preprocessing.ml_preprocessor import MLPreprocessor

def test_multi_timeframe_processing():
    print("Test du traitement multi-timeframes")
    print("="*50)
    
    # 1. Chargement des données
    print("\n1. Chargement des données")
    loader = HistoricalDataLoader()
    data_dict = loader.load_multi_timeframe_data()
    
    # Afficher les informations sur les données chargées
    print("\nRésumé des données chargées:")
    for tf, df in data_dict.items():
        print(f"\n{tf}:")
        print(f"- Période: {df.index.min()} à {df.index.max()}")
        print(f"- Nombre d'échantillons: {len(df)}")
        print(f"- Colonnes: {', '.join(df.columns)}")
        print(f"- Données manquantes: {df.isnull().sum().sum()}")
    
    # 2. Preprocessing
    print("\n2. Préparation des features")
    preprocessor = MLPreprocessor()
    
    # Préparer les features pour chaque timeframe
    processed_data = preprocessor.prepare_multi_timeframe_features(data_dict)
    
    # Choisir le timeframe de référence pour la cible (4h si 1d n'est pas disponible)
    reference_tf = '1d' if '1d' in data_dict else '4h'
    print(f"\nUtilisation de {reference_tf} comme timeframe de référence pour la cible")
    target = preprocessor.prepare_target(data_dict[reference_tf])
    
    # Afficher les informations sur les features
    print("\nRésumé des features générées:")
    for tf, features in processed_data.items():
        print(f"\n{tf}:")
        print(f"- Shape des features: {features.shape}")
        # Calculer les statistiques sur les valeurs numpy
        features_array = features.values
        print(f"- Moyenne: {np.mean(features_array):.4f}")
        print(f"- Écart-type: {np.std(features_array, axis=None):.4f}")
    
    # 3. Création des séquences
    print("\n3. Création des séquences")
    sequences = preprocessor.create_multi_timeframe_sequences(processed_data, target)
    
    # Afficher les informations sur les séquences
    print("\nRésumé des séquences:")
    for tf, (X, y) in sequences.items():
        print(f"\n{tf}:")
        print(f"- Shape X: {X.shape}")
        print(f"- Shape y: {y.shape}")
        print(f"- Nombre de séquences: {len(X)}")
    
    return data_dict, processed_data, sequences

if __name__ == "__main__":
    try:
        data_dict, processed_data, sequences = test_multi_timeframe_processing()
        print("\nTest terminé avec succès!")
        
        # Sauvegarder un résumé des résultats
        with open('data/processed/multi_timeframe_summary.txt', 'w') as f:
            f.write("Résumé du traitement multi-timeframes\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. Données brutes:\n")
            for tf, df in data_dict.items():
                f.write(f"\n{tf}:\n")
                f.write(f"- Échantillons: {len(df)}\n")
                f.write(f"- Période: {df.index.min()} à {df.index.max()}\n")
            
            f.write("\n2. Features:\n")
            for tf, features in processed_data.items():
                f.write(f"\n{tf}:\n")
                f.write(f"- Shape: {features.shape}\n")
            
            f.write("\n3. Séquences:\n")
            for tf, (X, y) in sequences.items():
                f.write(f"\n{tf}:\n")
                f.write(f"- Shape X: {X.shape}\n")
                f.write(f"- Shape y: {y.shape}\n")
        
    except Exception as e:
        print(f"\nErreur pendant le test: {str(e)}")
        raise 