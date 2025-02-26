from src.data_collection.historical_data import HistoricalDataCollector
from src.data_collection.marketstack_api import MarketstackAPI

# Test données historiques
collector = HistoricalDataCollector()
historical_data = collector.load_kaggle_data('1D')

# Test données temps réel
api = MarketstackAPI()
realtime_data = api.get_realtime_data()

print("Données historiques :", historical_data.head())
print("Données temps réel :", realtime_data.head()) 