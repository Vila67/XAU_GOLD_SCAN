import os
import pandas as pd
from src.data_collection.historical_data import HistoricalDataLoader

def test_data_loading():
    print("Test du chargement des données historiques")
    print("="*50)
    
    # Initialiser le loader
    loader = HistoricalDataLoader()
    
    # Vérifier les fichiers disponibles
    print("\nFichiers attendus:")
    for tf, filename in loader.timeframes.items():
        filepath = os.path.join(loader.data_dir, filename)
        exists = os.path.exists(filepath)
        print(f"- {filename}: {'✓' if exists else '✗'}")
    
    # Charger les données
    print("\nChargement des données...")
    data_dict = loader.load_multi_timeframe_data()
    
    if data_dict is None:
        print("Erreur: Aucune donnée n'a pu être chargée")
        return
    
    # Vérifier le format des données
    print("\nVérification du format des données:")
    for tf, df in data_dict.items():
        print(f"\n{tf}:")
        print(f"- Shape: {df.shape}")
        print(f"- Colonnes: {', '.join(df.columns)}")
        print(f"- Types des colonnes:")
        for col in df.columns:
            print(f"  * {col}: {df[col].dtype}")
        print(f"- Période: {df.index.min()} à {df.index.max()}")
        print(f"- Échantillon de données:")
        print(df.head(2))
    
    # Vérifier l'alignement des données
    print("\nVérification de l'alignement temporel:")
    start_dates = {tf: df.index.min() for tf, df in data_dict.items()}
    end_dates = {tf: df.index.max() for tf, df in data_dict.items()}
    
    print("\nDates de début:")
    for tf, date in start_dates.items():
        print(f"- {tf}: {date}")
    
    print("\nDates de fin:")
    for tf, date in end_dates.items():
        print(f"- {tf}: {date}")
    
    return data_dict

def test_historical_data_loading():
    """Test du chargement des données historiques"""
    loader = HistoricalDataLoader()
    
    # Test du chargement multi-timeframe
    data_dict = loader.load_multi_timeframe_data()
    
    # Vérifier la présence de tous les timeframes
    expected_timeframes = ['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    assert all(tf in data_dict for tf in expected_timeframes), "Timeframes manquants"
    
    # Vérifier la structure des données
    for tf, df in data_dict.items():
        print(f"\nVérification du timeframe {tf}:")
        assert isinstance(df, pd.DataFrame), f"Type incorrect pour {tf}"
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']), \
            f"Colonnes manquantes dans {tf}"
        print(f"Shape: {df.shape}")
        print(f"Période: {df.index[0]} à {df.index[-1]}")

if __name__ == "__main__":
    try:
        data = test_data_loading()
        if data is not None:
            print("\nTest terminé avec succès!")
            
            # Sauvegarder un résumé des résultats
            with open('data/processed/data_loading_summary.txt', 'w') as f:
                f.write("Résumé du chargement des données\n")
                f.write("="*50 + "\n\n")
                
                for tf, df in data.items():
                    f.write(f"\n{tf}:\n")
                    f.write(f"- Shape: {df.shape}\n")
                    f.write(f"- Période: {df.index.min()} à {df.index.max()}\n")
                    f.write(f"- Colonnes: {', '.join(df.columns)}\n")
                    f.write(f"- Données manquantes: {df.isnull().sum().sum()}\n")
    
    except Exception as e:
        print(f"\nErreur pendant le test: {str(e)}")
        raise 