import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.models.ml_model import MLModel
from src.preprocessing.ml_preprocessor import MLPreprocessor
import pandas as pd

def test_model_architectures():
    """Test des différentes architectures du modèle"""
    print("\n1. Test des architectures")
    print("="*50)
    
    # Créer des données de test
    X_train = np.random.random((1000, 15, 5))
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.random((200, 15, 5))
    y_val = np.random.randint(0, 3, 200)
    
    # Tester la sélection automatique
    model_auto = MLModel(input_shape=(15, 5), model_type='auto')
    best_type, best_score = model_auto.auto_select_model(X_train, y_train, X_val, y_val)
    
    print(f"\nMeilleure architecture: {best_type}")
    print(f"Score: {best_score:.3f}")
    
    # Tester chaque architecture individuellement
    for arch in ['transformer', 'lstm']:
        model = MLModel(input_shape=(15, 5), model_type=arch)
        model.train(X_train, y_train, X_val, y_val, epochs=2)
        
        # Vérifier les prédictions
        y_pred = model.predict(X_val)
        assert len(y_pred) == len(y_val), f"Erreur de dimension pour {arch}"

def test_trading_metrics():
    """Test des métriques spécifiques au trading"""
    print("\n2. Test des métriques de trading")
    print("="*50)
    
    model = MLModel(input_shape=(15, 5))
    
    # Créer des données de test
    y_true = np.array([1, -1, 0, 1, -1, 1, 0, 1])
    y_pred = np.array([1, -1, 1, 1, 0, 1, 0, -1])
    prices = np.array([100, 101, 99, 102, 98, 103, 101, 100])
    
    # Calculer les métriques
    metrics = model.calculate_trading_metrics(y_true, y_pred, prices)
    
    print("\nMétriques calculées:")
    for metric, value in metrics.items():
        print(f"• {metric}: {value}")
    
    # Vérifier les métriques essentielles
    assert 'win_rate' in metrics, "Win rate manquant"
    assert 'sharpe_ratio' in metrics, "Ratio Sharpe manquant"
    assert 'max_drawdown' in metrics, "Drawdown maximum manquant"

def test_confidence_scoring():
    """Test du système de score de confiance"""
    print("\n3. Test des scores de confiance")
    print("="*50)
    
    model = MLModel(input_shape=(15, 5))
    X_test = np.random.random((100, 15, 5))
    
    # Tester différents seuils de confiance
    thresholds = [0.5, 0.7, 0.9]
    for threshold in thresholds:
        predictions, scores, mask = model.predict_with_confidence(X_test, threshold)
        
        print(f"\nSeuil de confiance: {threshold}")
        print(f"• Prédictions retenues: {np.mean(mask):.2%}")
        print(f"• Score moyen: {np.mean(scores):.2%}")

def test_multi_timeframe():
    """Test du support multi-timeframes"""
    print("\n4. Test du multi-timeframe")
    print("="*50)
    
    # Créer des données multi-timeframes
    data_dict = {
        '5m': {'train': np.random.random((1000, 15, 5))},
        '15m': {'train': np.random.random((1000, 15, 5))},
        '1h': {'train': np.random.random((1000, 15, 5))},
        '4h': {'train': np.random.random((1000, 15, 5))}
    }
    y_train = np.random.randint(0, 3, 1000)
    
    # Créer et entraîner le modèle
    model = MLModel(input_shape=(15, 5))
    history = model.train_multi_timeframe(data_dict, y_train, epochs=2)
    
    print("\nVérification de l'entraînement multi-timeframe:")
    print(f"• Timeframes utilisés: {list(data_dict.keys())}")
    print(f"• Epochs complétées: {len(history.history['loss'])}")

def test_backtesting():
    """Test du système de backtesting"""
    print("\n5. Test du backtesting")
    print("="*50)
    
    # Créer des données historiques
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.random(1000) * 100,
        'High': np.random.random(1000) * 100,
        'Low': np.random.random(1000) * 100,
        'Close': np.random.random(1000) * 100,
        'Volume': np.random.random(1000) * 1000
    })
    
    # Créer et tester le modèle
    model = MLModel(input_shape=(15, 5))
    results = model.backtest_model(data)
    
    # Vérifier les résultats essentiels
    assert 'pnl' in results, "P&L manquant"
    assert 'roi' in results, "ROI manquant"
    assert 'trades' in results, "Analyse des trades manquante"

def test_ml_model():
    """Test complet du MLModel"""
    try:
        # 1. Test des architectures
        test_model_architectures()
        
        # 2. Test des métriques
        test_trading_metrics()
        
        # 3. Test des scores de confiance
        test_confidence_scoring()
        
        # 4. Test du multi-timeframe
        test_multi_timeframe()
        
        # 5. Test du backtesting
        test_backtesting()
        
        print("\n✅ Tous les tests ont réussi!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {str(e)}")
        raise

if __name__ == "__main__":
    test_ml_model() 