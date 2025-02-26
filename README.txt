TRADING BOT XAU/USD
==================

Description
----------
Bot de trading automatique spécialisé dans l'analyse multi-timeframes de l'or (XAU/USD).
Utilise l'apprentissage automatique et l'analyse technique.

Installation
-----------
1. Cloner le repository
2. Installer les dépendances : pip install -r requirements.txt
3. Configurer .env avec votre clé API Marketstack

Structure
---------
src/
    data_collection/   - Collecte des données
    preprocessing/     - Traitement des données
    features/         - Indicateurs techniques
    models/          - Modèles de prédiction
    visualization/    - Visualisation
tests/              - Tests unitaires
config/             - Configuration
requirements.txt    - Dépendances

Commandes Principales
-------------------
1. COLLECTE DE DONNÉES
   - Historique (une timeframe):
     python src/data_collection/historical_data.py --timeframe 5m
   
   - Toutes timeframes:
     python src/data_collection/historical_data.py --all-timeframes
   
   - Temps réel:
     python src/data_collection/marketstack_api.py

2. TRAITEMENT DES DONNÉES
   - Basique:
     python src/preprocessing/data_processor.py --input XAUUSD_5m.csv
   
   - Avancé:
     python src/preprocessing/data_processor.py --input XAUUSD_5m.csv --max-gap-multiplier 20.0

3. ANALYSE TECHNIQUE
   - Ajouter indicateurs:
     python src/features/technical_indicators.py --input XAUUSD_5m.csv
   
   - Générer rapport:
     python src/features/technical_indicators.py --report

4. TESTS
   - Tous les tests:
     pytest
   
   - Tests spécifiques:
     pytest test_data_processor.py
     pytest test_historical.py

Timeframes Disponibles
--------------------
5m  - 5 minutes
15m - 15 minutes
30m - 30 minutes
1h  - 1 heure
4h  - 4 heures
1d  - 1 jour
1w  - 1 semaine
1M  - 1 mois

Fonctionnalités
-------------
- Analyse multi-timeframes
- Gestion des gaps temporels
- Détection d'anomalies
- Indicateurs techniques
- Signaux de trading
- Équilibrage des données
- API Marketstack
- Validation des données
- Tests unitaires

Dépendances
----------
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.2
tensorflow >= 2.6.0
ta >= 0.7.0
kaggle == 1.5.16
requests == 2.31.0
python-dotenv == 1.0.0
matplotlib >= 3.7.1
seaborn >= 0.12.2

Notes
-----
- Données historiques disponibles de 2004 à 2024
- Optimisé pour l'or (XAU/USD)
- Minimum 500 échantillons requis par timeframe
- Gestion intelligente des gaps temporels
- Validation automatique des données 