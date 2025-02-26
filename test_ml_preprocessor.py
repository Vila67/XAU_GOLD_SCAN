import numpy as np
import pandas as pd
from src.preprocessing.ml_preprocessor import MLPreprocessor
from src.data_collection.historical_data import HistoricalDataLoader

def test_ml_preprocessor():
    """Test des fonctionnalités du MLPreprocessor"""
    print("Test du MLPreprocessor")
    print("="*50)
    
    # 1. Charger les données
    print("\n1. Chargement des données")
    loader = HistoricalDataLoader()
    data_dict = loader.load_multi_timeframe_data()
    
    # 2. Créer le preprocessor
    print("\n2. Initialisation du preprocessor")
    preprocessor = MLPreprocessor()
    
    # 3. Tester la préparation des données
    print("\n3. Test de la préparation des données")
    sequence_length = 15
    prediction_horizon = 12
    train_size = 0.8
    target_samples = 3000
    
    try:
        X_train, y_train, X_test, y_test = preprocessor.prepare_data_for_training(
            data_dict,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            train_size=train_size,
            target_samples=target_samples
        )
        
        # 4. Vérifier l'équilibre des données
        print("\n4. Vérification de l'équilibre des données")
        is_balanced = preprocessor.check_data_balance(y_train, y_test)
        
        if is_balanced:
            print("\n✅ Les données sont bien équilibrées")
        else:
            print("\n⚠️ Les données présentent un déséquilibre")
        
        # 5. Afficher les informations sur les features
        print("\n5. Information sur les features")
        for tf in X_train.keys():
            print(f"\nTimeframe {tf}:")
            print(f"Train shape: {X_train[tf].shape}")
            print(f"Test shape: {X_test[tf].shape}")
            print(f"Nombre de features: {X_train[tf].shape[-1]}")
            
            # Vérifier la normalisation
            print(f"Range train: [{X_train[tf].min():.3f}, {X_train[tf].max():.3f}]")
            print(f"Range test: [{X_test[tf].min():.3f}, {X_test[tf].max():.3f}]")
        
        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {str(e)}")
        raise

def test_feature_generation():
    """Test de la génération des features"""
    print("\nTest de la génération des features")
    print("="*50)
    
    # Charger un échantillon de données
    loader = HistoricalDataLoader()
    data_dict = loader.load_multi_timeframe_data()
    
    # Prendre un timeframe pour le test
    test_tf = '1h'
    test_data = data_dict[test_tf]
    
    preprocessor = MLPreprocessor()
    
    try:
        # Tester la préparation des features
        features_df = preprocessor.prepare_features(test_data)
        
        print(f"\nFeatures générées pour {test_tf}:")
        print(f"Shape: {features_df.shape}")
        print("\nAperçu des features:")
        print(features_df.head())
        
        # Vérifier la normalisation
        print("\nStatistiques des features:")
        print(f"Min: {features_df.min().min():.3f}")
        print(f"Max: {features_df.max().max():.3f}")
        print(f"Mean: {features_df.mean().mean():.3f}")
        
        return features_df
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test des features: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Tester le preprocessor
        X_train, y_train, X_test, y_test = test_ml_preprocessor()
        
        # Tester la génération de features
        features = test_feature_generation()
        
        print("\n✅ Tests terminés avec succès!")
        
    except Exception as e:
        print(f"\n❌ Tests échoués: {str(e)}")
        raise 