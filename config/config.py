import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MARKETSTACK_API_KEY = os.getenv('MARKETSTACK_API_KEY')
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

    # Créer les répertoires s'ils n'existent pas
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        os.makedirs(directory, exist_ok=True) 