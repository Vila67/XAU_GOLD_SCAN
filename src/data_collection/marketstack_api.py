import requests
import pandas as pd
from datetime import datetime
from config.config import Config
import os

class MarketstackAPI:
    def __init__(self):
        self.config = Config()
        self.base_url = "http://api.marketstack.com/v1"
        self.api_key = self.config.MARKETSTACK_API_KEY

    def get_realtime_data(self):
        """
        Récupère les données en temps réel pour XAU/USD
        """
        endpoint = f"{self.base_url}/intraday"
        params = {
            'access_key': self.api_key,
            'symbols': 'XAUUSD',
            'interval': '5min'
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Sauvegarder les données brutes
            filename = f"realtime_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(self.config.RAW_DATA_DIR, filename)
            
            df = pd.DataFrame(data['data'])
            df.to_csv(output_path, index=False)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête API: {str(e)}")
            return None 