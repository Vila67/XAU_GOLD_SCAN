from src.data_collection.historical_data import HistoricalDataCollector
from src.preprocessing.data_processor import GoldDataProcessor
from src.visualization.chart_plotter import GoldChartPlotter
import matplotlib.pyplot as plt
import os
import seaborn as sns

def ensure_directories():
    """Crée les dossiers nécessaires s'ils n'existent pas"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir

def test_gold_analysis():
    # 0. Préparer les dossiers
    processed_dir = ensure_directories()
    
    # Configuration du style
    sns.set_theme()
    plt.style.use('classic')
    
    # 1. Charger les données
    print("Chargement des données...")
    collector = HistoricalDataCollector()
    data = collector.load_kaggle_data('1D')
    
    if data is not None:
        # 2. Ajouter les indicateurs techniques
        print("\nAjout des indicateurs techniques...")
        processor = GoldDataProcessor()
        data_with_indicators = processor.add_technical_indicators(data)
        data_with_patterns = processor.add_price_patterns(data_with_indicators)
        
        # Afficher les premières lignes avec les indicateurs
        print("\nAperçu des données avec indicateurs :")
        print(data_with_patterns.head().to_string())
        
        # Afficher les statistiques des indicateurs
        print("\nStatistiques des indicateurs :")
        indicators = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']
        print(data_with_patterns[indicators].describe())
        
        # 3. Ajouter les signaux de trading
        data_with_signals = processor.add_trading_signals(data_with_patterns)
        
        # Créer les visualisations
        print("\nCréation des graphiques...")
        plotter = GoldChartPlotter()
        
        # Graphique des prix et volumes
        fig_price = plotter.plot_price_history(data_with_signals)
        
        # Graphique des indicateurs techniques
        fig_indicators = plotter.plot_technical_indicators(data_with_signals)
        
        # Sauvegarder les graphiques
        price_history_path = os.path.join(processed_dir, 'price_history.png')
        indicators_path = os.path.join(processed_dir, 'technical_indicators.png')
        
        print(f"\nSauvegarde des graphiques dans {processed_dir}...")
        fig_price.savefig(price_history_path, dpi=300, bbox_inches='tight')
        fig_indicators.savefig(indicators_path, dpi=300, bbox_inches='tight')
        
        print(f"\nAnalyse terminée ! Les graphiques ont été sauvegardés dans :")
        print(f"- {price_history_path}")
        print(f"- {indicators_path}")
        
        # Afficher les graphiques
        plt.show()
        
    else:
        print("Erreur : Impossible de charger les données")

if __name__ == "__main__":
    test_gold_analysis() 