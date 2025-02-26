import numpy as np
import pandas as pd
from src.preprocessing.ml_preprocessor import MLPreprocessor
from src.data_collection.historical_data import HistoricalDataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

def test_ml_preprocessor():
    """Test des fonctionnalités du MLPreprocessor"""
    print("Test du MLPreprocessor")
    print("="*50)
    
    # 1. Charger les données
    print("\n1. Chargement des données")
    loader = HistoricalDataLoader()
    data_dict = None  # Initialiser data_dict à None
    
    try:
        # Solution 1: Vérifier les colonnes
        data_dict = loader.load_multi_timeframe_data()
    except KeyError as e:
        if str(e) == "'Date'":
            print("⚠️ Erreur de colonne 'Date' - Tentative de correction...")
            
            # Solution 2: Essayer différents encodages et séparateurs
            try:
                data_dict = loader.load_multi_timeframe_data(encoding="latin-1", sep=";")
            except Exception:
                print("⚠️ Échec avec latin-1 et ; - Tentative avec autres paramètres...")
                
                # Solution 3: Essayer sans en-tête et détecter automatiquement
                try:
                    data_dict = loader.load_multi_timeframe_data(header=None)
                    
                    # Vérifier si les données sont chargées correctement
                    if '5m' in data_dict:
                        df_5m = data_dict['5m']
                        print("\nColonnes détectées dans 5m:")
                        print(df_5m.columns.tolist())
                        
                        # Renommer les colonnes si nécessaire
                        if 'Date' not in df_5m.columns:
                            df_5m.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                            data_dict['5m'] = df_5m
                            print("✅ Colonnes renommées avec succès")
                    
                except Exception as e:
                    print(f"❌ Échec du chargement: {str(e)}")
                    raise
    except Exception as e:
        print(f"❌ Erreur inattendue: {str(e)}")
        raise
    
    # Vérifier si les données ont été chargées
    if data_dict is None:
        raise ValueError("Impossible de charger les données après toutes les tentatives")
    
    # 2. Créer le preprocessor
    print("\n2. Initialisation du preprocessor")
    preprocessor = MLPreprocessor()
    
    # 3. Test de la préparation des données
    print("\n3. Test de la préparation des données")
    sequence_length = 15
    prediction_horizon = 12
    train_size = 0.8
    target_samples = 3000
    
    try:
        def stratified_kfold_split(X_dict, y, train_size=0.8, n_splits=5):
            """Split stratifié avec KFold pour une meilleure répartition"""
            from sklearn.model_selection import StratifiedKFold
            
            print("\nApplication du split stratifié KFold:")
            print("Distribution initiale des classes:")
            for class_label in np.unique(y):
                count = np.sum(y == class_label)
                ratio = count / len(y) * 100
                print(f"Classe {class_label}: {count} échantillons ({ratio:.2f}%)")
            
            # Utiliser StratifiedKFold pour une meilleure répartition
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Sélectionner le meilleur split
            best_split = None
            min_diff = float('inf')
            
            # Créer un array d'indices de la taille de y
            indices = np.arange(len(y))
            
            for train_idx, test_idx in skf.split(indices, y):
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                # Calculer les différences de distribution
                train_dist = np.array([np.sum(y_train_fold == c) / len(y_train_fold) 
                                     for c in [-1, 0, 1]])
                test_dist = np.array([np.sum(y_test_fold == c) / len(y_test_fold) 
                                     for c in [-1, 0, 1]])
                
                diff = np.max(np.abs(train_dist - test_dist))
                
                if diff < min_diff:
                    min_diff = diff
                    best_split = (train_idx, test_idx)
            
            if best_split is None:
                raise ValueError("Impossible de trouver un split valide")
            
            # Créer les dictionnaires X_train et X_test
            X_train = {}
            X_test = {}
            
            # Diviser les données pour chaque timeframe
            timeframes = ['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
            for tf in timeframes:
                if tf in X_dict:
                    # Vérifier que les indices sont valides
                    max_idx = len(X_dict[tf]['X'])
                    train_indices = best_split[0][best_split[0] < max_idx]
                    test_indices = best_split[1][best_split[1] < max_idx]
                    
                    # Assigner les données avec les indices valides
                    X_train[tf] = X_dict[tf]['X'][train_indices]
                    X_test[tf] = X_dict[tf]['X'][test_indices]
                    
                    print(f"\nTimeframe {tf}:")
                    print(f"Taille totale: {max_idx}")
                    print(f"Train: {len(train_indices)} échantillons")
                    print(f"Test: {len(test_indices)} échantillons")
            
            return X_train, X_test, y[best_split[0]], y[best_split[1]]
        
        # Préparer les données avec le nouveau split stratifié
        sequences_dict = preprocessor.create_sequences(
            data_dict,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_samples=target_samples
        )
        
        # Extraire X_combined et y_combined du dictionnaire
        y_combined = np.concatenate([sequences_dict[tf]['y'] for tf in sequences_dict], axis=0)
        
        # Appliquer le split stratifié amélioré
        X_train, X_test, y_train, y_test = stratified_kfold_split(
            sequences_dict,
            y_combined,
            train_size=train_size
        )
        
        # Vérifier la distribution après split
        print("\nDistribution après split stratifié:")
        for set_name, y_data in [("Train", y_train), ("Test", y_test)]:
            print(f"\n{set_name}:")
            for class_label in np.unique(y_combined):
                count = np.sum(y_data == class_label)
                ratio = count / len(y_data) * 100
                print(f"Classe {class_label}: {count} échantillons ({ratio:.2f}%)")
        
        # Appliquer SMOTE si nécessaire pour la classe neutre
        if np.sum(y_train == 0) < 40:  # Minimum de 40 échantillons neutres
            print("\nApplication de SMOTE pour la classe neutre:")
            try:
                # Préparer les données pour SMOTE
                # Calculer d'abord le nombre total de features
                total_features = sum(np.prod(X_train[tf].shape[1:]) for tf in X_train)
                X_train_flat = np.zeros((len(y_train), total_features))
                
                # Remplir X_train_flat avec les données de chaque timeframe
                current_pos = 0
                for tf in X_train:
                    n_features = np.prod(X_train[tf].shape[1:])
                    X_train_flat[:, current_pos:current_pos+n_features] = X_train[tf].reshape(len(y_train), -1)
                    current_pos += n_features
                
                # Calculer le nombre cible d'échantillons neutres
                target_neutral = max(40, int(0.05 * len(y_train)))  # Au moins 5% ou 40 échantillons
                
                smote = SMOTE(
                    sampling_strategy={0: target_neutral},
                    random_state=42,
                    k_neighbors=min(5, np.sum(y_train == 0) - 1)
                )
                
                X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)
                
                # Reconstruire le dictionnaire X_train
                X_train_new = {}
                current_pos = 0
                for tf in X_train:
                    shape = X_train[tf].shape
                    n_features = np.prod(shape[1:])
                    X_train_new[tf] = X_resampled[:, current_pos:current_pos+n_features].reshape(-1, *shape[1:])
                    current_pos += n_features
                
                X_train = X_train_new
                y_train = y_resampled
                
                print("\nDistribution après SMOTE:")
                for class_label in np.unique(y_train):
                    count = np.sum(y_train == class_label)
                    ratio = count / len(y_train) * 100
                    print(f"Classe {class_label}: {count} échantillons ({ratio:.2f}%)")
                
            except Exception as e:
                print(f"⚠️ Erreur lors de l'application de SMOTE: {str(e)}")
        
        # 4. Vérifier l'équilibre des données de manière plus détaillée
        print("\n4. Vérification détaillée de l'équilibre des données")
        is_balanced = preprocessor.check_data_balance(y_train, y_test)
        
        # Ajouter une analyse plus détaillée des proportions
        def print_class_distribution(y, set_name):
            unique, counts = np.unique(y, return_counts=True)
            dist = dict(zip(unique, counts))
            total = sum(counts)
            print(f"\nDistribution des classes ({set_name}):")
            for classe, count in dist.items():
                percentage = (count/total) * 100
                print(f"Classe {classe}: {count} ({percentage:.2f}%)")
        
        print_class_distribution(y_train, "Train")
        print_class_distribution(y_test, "Test")
        
        # Calculer le déséquilibre entre train et test de manière plus robuste
        def get_class_distribution(y):
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            dist = {int(class_): count/total for class_, count in zip(unique, counts)}
            return dist
        
        train_dist = get_class_distribution(y_train)
        test_dist = get_class_distribution(y_test)
        
        # S'assurer que toutes les classes sont présentes dans les deux distributions
        all_classes = sorted(set(train_dist.keys()) | set(test_dist.keys()))
        for class_ in all_classes:
            if class_ not in train_dist:
                train_dist[class_] = 0
            if class_ not in test_dist:
                test_dist[class_] = 0
        
        # Calculer le déséquilibre maximal
        max_imbalance = max(abs(train_dist[class_] - test_dist[class_]) * 100 
                           for class_ in all_classes)
        
        print(f"\nDéséquilibre maximal entre train et test: {max_imbalance:.2f}%")
        
        # Afficher les différences par classe
        print("\nDifférences par classe (Test - Train):")
        for class_ in all_classes:
            diff = (test_dist[class_] - train_dist[class_]) * 100
            print(f"Classe {class_}: {diff:+.2f}%")
        
        if max_imbalance > 5:
            print("⚠️ Déséquilibre significatif détecté (>5%)")
        if is_balanced:
            print("✅ Les données sont globalement équilibrées")
        else:
            print("⚠️ Les données présentent un déséquilibre")
        
        # 5. Afficher les informations sur les features
        print("\n5. Information sur les features")
        for tf in sequences_dict:
            print(f"\nTimeframe {tf}:")
            print(f"Train shape: {X_train[tf].shape}")
            print(f"Test shape: {X_test[tf].shape}")
            print(f"Nombre de features: {X_train[tf].shape[-1]}")
            
            # Vérifier la normalisation
            print(f"Range train: [{X_train[tf].min():.3f}, {X_train[tf].max():.3f}]")
            print(f"Range test: [{X_test[tf].min():.3f}, {X_test[tf].max():.3f}]")
        
        # Modifier la fonction d'analyse des ratios d'alignement
        print("\n6. Analyse des ratios d'alignement")
        target_ratio = 0.20  # ratio cible pour les timeframes longs
        
        def analyze_timeframe_ratios(X_data):
            base_samples = max([data.shape[0] for data in X_data.values()])
            ratios = {}
            long_timeframes = ['1w', '1M']  # Timeframes longs à surveiller particulièrement
            
            print("\nAnalyse générale:")
            for tf, data in X_data.items():
                ratio = data.shape[0] / base_samples
                ratios[tf] = ratio
                status = "✅" if ratio >= target_ratio else "⚠️"
                print(f"{status} {tf}: {ratio:.3f} ({data.shape[0]} échantillons)")
                
                if ratio < target_ratio:
                    current_factor = 1.0
                    suggested_factor = min(3.0, target_ratio / ratio)  # Limité à 3x max
                    additional_samples = int((suggested_factor - 1) * data.shape[0])
                    
                    if tf in long_timeframes:
                        print(f"   📊 Timeframe long détecté: {tf}")
                        print(f"   → Ratio actuel: {ratio:.3f}")
                        print(f"   → Facteur d'upsampling suggéré: {suggested_factor:.2f}x")
                        print(f"   → Échantillons supplémentaires: {additional_samples}")
                        if suggested_factor > 2.5:
                            print("   ⚠️ Attention: Facteur élevé, considérer une approche progressive")
                    else:
                        print(f"   → Suggestion: Augmenter le facteur d'upsampling de {suggested_factor:.2f}x")
            
            # Analyse spécifique des timeframes longs
            print("\nRésumé des timeframes longs:")
            long_tf_stats = {tf: ratios[tf] for tf in long_timeframes if tf in ratios}
            if long_tf_stats:
                avg_long_ratio = sum(long_tf_stats.values()) / len(long_tf_stats)
                print(f"Ratio moyen des timeframes longs: {avg_long_ratio:.3f}")
                if avg_long_ratio < target_ratio:
                    print(f"⚠️ Les timeframes longs sont sous-représentés (cible: {target_ratio})")
                    print("Suggestions d'optimisation:")
                    for tf, ratio in long_tf_stats.items():
                        optimal_factor = min(3.0, target_ratio / ratio)
                        print(f"- {tf}: Utiliser un facteur d'upsampling de {optimal_factor:.2f}x")
            
            return ratios
        
        print("\nRatios d'alignement (Train):")
        train_ratios = analyze_timeframe_ratios(X_train)
        print("\nRatios d'alignement (Test):")
        test_ratios = analyze_timeframe_ratios(X_test)
        
        # Améliorer l'analyse de la classe neutre avec focus sur l'absence dans le train set
        print("\n7. Analyse détaillée de la classe neutre")
        def analyze_neutral_class(y_data, set_name):
            total = len(y_data)
            neutral_samples = np.sum(y_data == 0)
            neutral_ratio = (neutral_samples / total) * 100
            
            print(f"\nClasse neutre ({set_name}):")
            print(f"Nombre d'échantillons: {neutral_samples}")
            print(f"Ratio actuel: {neutral_ratio:.2f}%")
            
            if neutral_samples == 0:
                print("⚠️ CRITIQUE: Classe neutre absente!")
                # Calculer le nombre d'échantillons nécessaires pour atteindre ~5%
                target_samples = int(0.05 * total)
                print(f"→ Nécessité d'ajouter {target_samples} échantillons")
                print("→ Suggestions:")
                print("   1. Utiliser SMOTE ou ADASYN pour générer des échantillons synthétiques")
                print("   2. Transférer des échantillons du test set (si disponible)")
                print("   3. Réévaluer la stratégie de split train/test")
            elif neutral_ratio < 5:
                print("⚠️ Classe neutre sous-représentée")
                target_samples = int(0.05 * total)
                additional_needed = target_samples - neutral_samples
                print(f"→ Ajouter {additional_needed} échantillons pour atteindre 5%")
                if set_name == "Test" and neutral_ratio > 0:
                    print("→ Possibilité de transférer des échantillons vers le train set")
            elif 5 <= neutral_ratio <= 15:
                print("✅ Ratio acceptable (5-15%)")
            else:
                print("⚠️ Classe neutre sur-représentée")
                suggested_reduction = int(neutral_samples - (0.15 * total))
                print(f"→ Réduire de {suggested_reduction} échantillons pour atteindre 15%")
            
            return neutral_ratio, neutral_samples
        
        train_neutral_ratio, train_neutral_samples = analyze_neutral_class(y_train, "Train")
        test_neutral_ratio, test_neutral_samples = analyze_neutral_class(y_test, "Test")
        
        # Analyse spécifique du transfert possible entre test et train
        if train_neutral_samples == 0 and test_neutral_samples > 0:
            print("\nPlan de rééquilibrage suggéré:")
            safe_transfer = min(97, int(test_neutral_samples * 0.4))  # Limite à 40% max du test set
            remaining_needed = 97 - safe_transfer
            if remaining_needed > 0:
                print(f"2. Générer {remaining_needed} échantillons synthétiques additionnels")
            print("\nÉtapes de mise en œuvre:")
            print("1. Utiliser un split stratifié pour la division initiale")
            print("2. Appliquer SMOTE/ADASYN après le split pour les échantillons synthétiques")
            print("3. Valider la qualité des échantillons générés")
        
        # Calculer la différence de ratio entre train et test
        neutral_diff = abs(train_neutral_ratio - test_neutral_ratio)
        print(f"\nDifférence de ratio entre train et test: {neutral_diff:.2f}%")
        if neutral_diff > 2:
            print("⚠️ Différence significative entre train et test")
            print("→ Suggestion: Ajuster les ratios d'oversampling pour maintenir une distribution similaire")
        
        # Ajouter une analyse de la différence de distribution train/test
        print("\n8. Analyse de la différence de distribution train/test")
        def analyze_distribution_split(y_train, y_test):
            train_dist = get_class_distribution(y_train)
            test_dist = get_class_distribution(y_test)
            all_classes = sorted(set(train_dist.keys()) | set(test_dist.keys()))
            
            print("\nComparaison détaillée des distributions:")
            print("Classe  |  Train   |   Test   | Différence")
            print("--------+----------+----------+-----------")
            
            max_diff_class = None
            max_diff = 0
            
            for class_ in all_classes:
                train_pct = train_dist.get(class_, 0) * 100
                test_pct = test_dist.get(class_, 0) * 100
                diff = test_pct - train_pct
                
                if abs(diff) > abs(max_diff):
                    max_diff = diff
                    max_diff_class = class_
                
                status = "⚠️" if abs(diff) > 2 else "✅"
                print(f"{class_:^7} | {train_pct:>6.2f}% | {test_pct:>6.2f}% | {diff:>+7.2f}% {status}")
            
            print(f"\nClasse la plus déséquilibrée: {max_diff_class} (écart de {max_diff:+.2f}%)")
            
            # Suggestions d'amélioration
            if abs(max_diff) > 2:
                print("\nSuggestions d'amélioration:")
                print("1. Ajustement du split train/test:")
                if max_diff > 0:
                    print(f"   → Augmenter la proportion de classe {max_diff_class} dans le train set")
                    suggested_adjustment = abs(max_diff) / 2
                    print(f"   → Transférer ~{suggested_adjustment:.1f}% des échantillons du test vers le train")
                else:
                    print(f"   → Augmenter la proportion de classe {max_diff_class} dans le test set")
                    suggested_adjustment = abs(max_diff) / 2
                    print(f"   → Transférer ~{suggested_adjustment:.1f}% des échantillons du train vers le test")
                
                print("\n2. Stratégies de rééchantillonnage:")
                print("   → Utiliser un split stratifié pour maintenir les proportions")
                print("   → Appliquer l'oversampling après le split pour éviter les fuites")
                
                # Calcul du ratio de rééchantillonnage optimal
                target_ratio = (train_dist.get(max_diff_class, 0) + test_dist.get(max_diff_class, 0)) / 2
                print(f"\n3. Ratio cible suggéré pour la classe {max_diff_class}: {target_ratio*100:.2f}%")
            
            return max_diff
        
        max_distribution_diff = analyze_distribution_split(y_train, y_test)
        
        # Ajouter l'analyse détaillée de l'asymétrie train/test
        print("\n9. Analyse détaillée de l'asymétrie train/test")
        def analyze_class_asymmetry(y_train, y_test):
            print("\nAnalyse de l'asymétrie par classe:")
            print("="*50)
            
            # Calculer les distributions
            train_dist = get_class_distribution(y_train)
            test_dist = get_class_distribution(y_test)
            all_classes = sorted(set(train_dist.keys()) | set(test_dist.keys()))
            
            class_labels = {
                -1: "baisse",
                0: "neutre",
                1: "hausse"
            }
            
            # Analyser chaque classe
            asymmetry_stats = {}
            for class_ in all_classes:
                train_pct = train_dist.get(class_, 0) * 100
                test_pct = test_dist.get(class_, 0) * 100
                diff = test_pct - train_pct
                
                label = class_labels.get(class_, str(class_))
                print(f"\nClasse {class_} ({label}):")
                print(f"Train: {train_pct:.2f}% | Test: {test_pct:.2f}% | Écart: {diff:+.2f}%")
                
                # Suggestions spécifiques par classe
                if abs(diff) > 2:
                    print("⚠️ Asymétrie significative détectée")
                    if diff > 0:
                        print(f"→ La classe est sur-représentée dans le test set ({diff:+.2f}%)")
                        target_adjustment = abs(diff) / 2
                        print(f"→ Suggestion: Transférer {target_adjustment:.1f}% vers le train set")
                    else:
                        print(f"→ La classe est sous-représentée dans le test set ({diff:+.2f}%)")
                        target_adjustment = abs(diff) / 2
                        print(f"→ Suggestion: Augmenter de {target_adjustment:.1f}% dans le test set")
                else:
                    print("✅ Distribution relativement équilibrée")
                
                asymmetry_stats[class_] = diff
            
            # Suggestions globales
            print("\nSuggestions d'amélioration globales:")
            print("="*50)
            print("1. Stratégie de split:")
            print("   → Implémenter un StratifiedKFold pour le split train/test")
            print("   → Utiliser un random_state fixe pour la reproductibilité")
            
            print("\n2. Stratégie d'oversampling:")
            print("   → Appliquer l'oversampling séparément sur train et test")
            print("   → Ajuster les ratios d'oversampling par classe:")
            for class_ in all_classes:
                diff = asymmetry_stats[class_]
                if abs(diff) > 1:
                    label = class_labels.get(class_, str(class_))
                    if diff > 0:
                        print(f"     • Classe {class_} ({label}): Augmenter de {abs(diff/2):.1f}% dans le train")
                    else:
                        print(f"     • Classe {class_} ({label}): Augmenter de {abs(diff/2):.1f}% dans le test")
            
            return asymmetry_stats
        
        asymmetry_stats = analyze_class_asymmetry(y_train, y_test)
        
        # Ajouter l'analyse du bruit et de l'upsampling contrôlé
        print("\n10. Analyse du bruit et upsampling contrôlé")
        def analyze_noise_upsampling(X_data, timeframes=['1M', '1w', '1d']):
            print("\nAnalyse du bruit et suggestions d'upsampling:")
            print("="*50)
            
            base_samples = max([data.shape[0] for data in X_data.values()])
            noise_configs = {}
            
            # Paramètres globaux plus conservateurs
            max_global_factor = 2.0  # Limite globale à 2x
            
            for tf in timeframes:
                if tf not in X_data:
                    continue
                    
                data = X_data[tf]
                current_samples = data.shape[0]
                ratio = current_samples / base_samples
                
                print(f"\nTimeframe {tf}:")
                print(f"Échantillons actuels: {current_samples}")
                print(f"Ratio actuel: {ratio:.3f}")
                
                # Calculer le facteur d'upsampling optimal avec nouvelle limite
                target_ratio = min(0.20, ratio * 2)  # Maximum 2x ou 20%
                upsampling_factor = min(max_global_factor, target_ratio / ratio)
                
                # Configuration du bruit plus conservative
                if tf == '1M':
                    noise_scale = 0.0005  # Réduit de moitié
                    max_factor = 1.5      # Plus conservateur
                elif tf == '1w':
                    noise_scale = 0.001   # Réduit de moitié
                    max_factor = 1.75     # Plus conservateur
                else:
                    noise_scale = 0.002   # Réduit de plus de moitié
                    max_factor = 2.0      # Limite standard
                
                # Appliquer les limites plus strictes
                upsampling_factor = min(upsampling_factor, max_factor)
                target_samples = int(current_samples * upsampling_factor)
                
                noise_configs[tf] = {
                    'factor': upsampling_factor,
                    'noise_scale': noise_scale,
                    'target_samples': target_samples
                }
                
                print("\nSuggestions d'upsampling (conservatrices):")
                print(f"→ Facteur suggéré: {upsampling_factor:.2f}x")
                print(f"→ Échantillons cibles: {target_samples}")
                print(f"→ Échelle de bruit réduite: {noise_scale}")
                
                if upsampling_factor > 1.0:
                    additional = target_samples - current_samples
                    print(f"→ Échantillons à générer: {additional}")
                    print("\nRecommandations de bruit (conservatrices):")
                    print(f"• Utiliser une distribution normale réduite: N(0, {noise_scale})")
                    print("• Appliquer le bruit progressivement")
                    print("• Valider chaque batch d'échantillons générés")
                    
                    if upsampling_factor >= 1.5:
                        print("\n⚠️ Attention: Facteur d'upsampling significatif")
                        print("→ Valider avec cross-validation")
                        print("→ Considérer une approche progressive:")
                        print(f"   1. Commencer avec {upsampling_factor/2:.2f}x")
                        print("   2. Évaluer les performances")
                        print("   3. Augmenter si nécessaire")
                
                # Ajouter des métriques de qualité
                if 'original_std' in locals():
                    quality_ratio = noise_scale / original_std
                    print(f"\nRatio bruit/variance: {quality_ratio:.3f}")
                    if quality_ratio > 0.1:
                        print("⚠️ Niveau de bruit potentiellement élevé")
            
            return noise_configs
        
        print("\nAnalyse du train set:")
        train_noise_configs = analyze_noise_upsampling(X_train)
        print("\nAnalyse du test set:")
        test_noise_configs = analyze_noise_upsampling(X_test)
        
        # Ajouter l'implémentation SMOTE pour la classe neutre
        print("\n11. Rééquilibrage de la classe neutre avec SMOTE")
        def apply_smote_resampling(X_train, y_train, X_test, y_test):
            print("\nApplication de SMOTE pour la classe neutre:")
            print("="*50)
            
            # Analyser la situation actuelle
            train_neutral = np.sum(y_train == 0)
            test_neutral = np.sum(y_test == 0)
            test_neutral_ratio = test_neutral / len(y_test)
            
            print(f"\nÉtat initial:")
            print(f"• Train: {train_neutral} échantillons neutres")
            print(f"• Test: {test_neutral} échantillons neutres ({test_neutral_ratio*100:.2f}%)")
            
            if train_neutral == 0:
                # Calculer le nombre d'échantillons à transférer
                safe_transfer = min(16, int(test_neutral * 0.4))
                remaining_needed = 81
                
                print(f"\nPlan de rééquilibrage:")
                print(f"1. Transfert test → train: {safe_transfer} échantillons")
                print(f"2. Génération SMOTE: {remaining_needed} échantillons")
                
                # Préparer les indices pour le transfert
                neutral_indices = np.where(y_test == 0)[0]
                transfer_indices = np.random.choice(neutral_indices, safe_transfer, replace=False)
                
                # Effectuer le transfert pour chaque timeframe
                X_train_new = {}
                X_test_new = {}
                
                print("\nTransfert des échantillons test → train...")
                for tf in X_train.keys():
                    # Transférer les échantillons du test vers le train
                    X_train_new[tf] = np.concatenate([X_train[tf], X_test[tf][transfer_indices]])
                    
                    # Mettre à jour le test set
                    mask = np.ones(len(y_test), dtype=bool)
                    mask[transfer_indices] = False
                    X_test_new[tf] = X_test[tf][mask]
                
                # Mettre à jour les labels
                y_train_new = np.concatenate([y_train, y_test[transfer_indices]])
                y_test_new = y_test[mask]
                
                # Appliquer SMOTE
                print("\nApplication de SMOTE...")
                target_samples = train_neutral + safe_transfer + remaining_needed
                sampling_strategy = {0: target_samples}
                
                try:
                    # Préparer les données pour SMOTE (combiner tous les timeframes)
                    X_train_combined = np.concatenate([X_train_new[tf].reshape(X_train_new[tf].shape[0], -1) 
                                                     for tf in X_train_new.keys()], axis=1)
                    
                    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train_new)
                    
                    # Répartir les données resamplees dans les timeframes
                    feature_start = 0
                    X_train_final = {}
                    for tf in X_train_new.keys():
                        original_shape = X_train_new[tf].shape
                        feature_count = np.prod(original_shape[1:])  # Multiplier toutes les dimensions sauf la première
                        features = X_train_resampled[:, feature_start:feature_start + feature_count]
                        X_train_final[tf] = features.reshape(features.shape[0], *original_shape[1:])
                        feature_start += feature_count
                    
                    # Vérifier les résultats
                    final_train_neutral = np.sum(y_train_resampled == 0)
                    final_test_neutral = np.sum(y_test_new == 0)
                    
                    print("\nRésultats du rééquilibrage:")
                    print(f"Train: {final_train_neutral} échantillons neutres")
                    print(f"Test: {final_test_neutral} échantillons neutres")
                    
                    return X_train_final, y_train_resampled, X_test_new, y_test_new
                    
                except Exception as e:
                    print(f"❌ Erreur lors de l'application de SMOTE: {str(e)}")
                    return X_train, y_train, X_test, y_test
            else:
                print("✅ Classe neutre déjà présente dans le train set")
                return X_train, y_train, X_test, y_test
        
        # Appliquer le rééquilibrage
        X_train, y_train, X_test, y_test = apply_smote_resampling(X_train, y_train, X_test, y_test)
        
        # Ajouter le rééquilibrage de la classe hausse (1)
        print("\n12. Rééquilibrage de la classe hausse (1)")
        def balance_positive_class(X_train, y_train, X_test, y_test):
            """Équilibre la classe +1 entre train et test"""
            print("\nÉquilibrage de la classe +1:")
            print("="*50)
            
            # Calculer les ratios actuels
            train_pos = np.sum(y_train == 1) / len(y_train) * 100
            test_pos = np.sum(y_test == 1) / len(y_test) * 100
            diff = train_pos - test_pos
            
            print(f"Distribution initiale classe +1:")
            print(f"Train: {train_pos:.2f}%")
            print(f"Test: {test_pos:.2f}%")
            print(f"Différence: {diff:.2f}%")
            
            if diff > 2:  # Si déséquilibre significatif
                # Calculer le nombre d'échantillons à transférer
                n_samples_to_transfer = int((diff/2) * len(y_train) / 100)
                print(f"\nTransfert de {n_samples_to_transfer} échantillons de Train vers Test")
                
                # Sélectionner les indices à transférer
                pos_indices = np.where(y_train == 1)[0]
                transfer_indices = np.random.choice(pos_indices, n_samples_to_transfer, replace=False)
                
                # Effectuer le transfert
                X_train_new = {}
                X_test_new = {}
                
                for tf in X_train.keys():
                    # Ajouter au test
                    X_test_new[tf] = np.concatenate([X_test[tf], X_train[tf][transfer_indices]])
                    # Retirer du train
                    mask = np.ones(len(y_train), dtype=bool)
                    mask[transfer_indices] = False
                    X_train_new[tf] = X_train[tf][mask]
                
                y_test_new = np.concatenate([y_test, y_train[transfer_indices]])
                y_train_new = y_train[mask]
                
                # Vérifier les nouveaux ratios
                new_train_pos = np.sum(y_train_new == 1) / len(y_train_new) * 100
                new_test_pos = np.sum(y_test_new == 1) / len(y_test_new) * 100
                new_diff = new_train_pos - new_test_pos
                
                print(f"\nNouvelle distribution classe +1:")
                print(f"Train: {new_train_pos:.2f}%")
                print(f"Test: {new_test_pos:.2f}%")
                print(f"Différence: {new_diff:.2f}%")
                
                return X_train_new, y_train_new, X_test_new, y_test_new
            
            return X_train, y_train, X_test, y_test
        
        # Appliquer le rééquilibrage de la classe hausse
        X_train, y_train, X_test, y_test = balance_positive_class(X_train, y_train, X_test, y_test)
        
        # Préparer les données pour la réduction de dimensionnalité
        print("\nPréparation des données pour la réduction de dimensionnalité")
        X_combined = prepare_data_for_reduction(X_train)

        # Optimiser la réduction de dimensionnalité
        print("\nApplication de la réduction de dimensionnalité:")
        X_reduced, reducer = optimize_dimensionality_reduction(X_combined)

        # Vérifier les résultats
        print("\nRésultats de la réduction:")
        print(f"• Dimensions initiales: {X_combined.shape}")
        print(f"• Dimensions réduites: {X_reduced.shape}")

        # Calculer les métriques de qualité
        variances = np.var(X_reduced, axis=0)
        print("\nAnalyse des composantes:")
        print(f"• Variance minimale: {np.min(variances):.2e}")
        print(f"• Variance médiane: {np.median(variances):.2e}")
        print(f"• Composantes significatives (var >= 1e-5): {np.sum(variances >= 1e-5)}")
        
        # Ajouter l'équilibrage de la classe neutre
        print("\nApplication de l'équilibrage de la classe neutre")
        X_train, y_train = balance_neutral_class(X_train, y_train)
        
        # Ajouter l'équilibrage des timeframes longs
        print("\nApplication de l'équilibrage des timeframes longs")
        sequences_dict = balance_long_timeframes(sequences_dict, min_ratio=0.01)
        
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
        
        # Gérer les valeurs manquantes de manière moderne
        if isinstance(features_df, pd.DataFrame):
            # Remplacer fillna par ffill/bfill pour les DataFrames
            features_df = features_df.ffill().bfill()
        elif isinstance(features_df, dict):
            # Traiter chaque DataFrame dans le dictionnaire
            for key in features_df:
                if isinstance(features_df[key], pd.DataFrame):
                    features_df[key] = features_df[key].ffill().bfill()
        
        print(f"\nFeatures générées pour {test_tf}:")
        print(f"Shape: {features_df.shape}")
        print("\nAperçu des features:")
        print(features_df.head())
        
        # Vérifier la normalisation
        print("\nStatistiques des features:")
        if isinstance(features_df, pd.DataFrame):
            # Calculer les statistiques descriptives
            stats = features_df.describe()
            print(f"Minimum: {stats.loc['min'].min():.3f}")
            print(f"Maximum: {stats.loc['max'].max():.3f}")
            print(f"Moyenne: {stats.loc['mean'].mean():.3f}")
            print(f"Écart-type: {stats.loc['std'].mean():.3f}")
            
            # Afficher des informations supplémentaires
            print("\nDistribution des valeurs:")
            print(f"25%: {stats.loc['25%'].mean():.3f}")
            print(f"50%: {stats.loc['50%'].mean():.3f}")
            print(f"75%: {stats.loc['75%'].mean():.3f}")
            
            # Vérifier les valeurs manquantes
            null_count = features_df.isnull().sum().sum()
            if null_count > 0:
                print(f"\n⚠️ {null_count} valeurs manquantes détectées")
            else:
                print("\n✅ Aucune valeur manquante")
        
        return features_df
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test des features: {str(e)}")
        raise

def optimize_dimensionality_reduction(X_train_filtered, method='pca', min_explained_variance=0.90, recursion_count=0):
    """Optimise la réduction de dimensionnalité avec PCA ou KernelPCA"""
    # Limiter la récursion
    max_recursions = 2
    if recursion_count >= max_recursions:
        print("\n⚠️ Nombre maximum de tentatives atteint")
        print("→ Utilisation de PCA conservative")
        # Fallback sur PCA simple avec paramètres conservateurs
        n_components = min(50, X_train_filtered.shape[1], X_train_filtered.shape[0])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X_train_filtered), pca
    
    print(f"\nOptimisation de la réduction de dimensionnalité (tentative {recursion_count + 1}/{max_recursions}):")
    print(f"Méthode: {method.upper()}")
    
    try:
        if method == 'pca':
            # Commencer avec une analyse PCA complète
            pca_analyzer = PCA()
            pca_analyzer.fit(X_train_filtered)
            
            # Analyser la courbe de variance expliquée
            cumsum = np.cumsum(pca_analyzer.explained_variance_ratio_)
            
            # Trouver le nombre optimal de composantes
            n_components = np.argmax(cumsum >= min_explained_variance) + 1
            total_components = min(X_train_filtered.shape[1], X_train_filtered.shape[0])
            
            print(f"\nAnalyse de la variance expliquée:")
            print(f"• Composantes totales possibles: {total_components}")
            print(f"• Variance expliquée cumulée:")
            for threshold in [0.80, 0.85, 0.90, 0.95, 0.99]:
                n = min(np.argmax(cumsum >= threshold) + 1, total_components)
                print(f"  {threshold:.0%}: {n} composantes ({cumsum[n-1]:.2%} de variance)")
            
            # Limiter la réduction pour préserver l'information
            max_reduction = 0.5  # Ne pas réduire de plus de 50%
            min_components = min(int(total_components * max_reduction), total_components)
            n_components = min(max(n_components, min_components), total_components)
            
            print(f"\nParamètres retenus:")
            print(f"• Composantes: {n_components} ({n_components/total_components:.1%} des features)")
            print(f"• Variance expliquée: {cumsum[n_components-1]:.2%}")
            
            # Appliquer PCA avec les paramètres optimisés
            pca = PCA(n_components=n_components, svd_solver='full')
            X_reduced = pca.fit_transform(X_train_filtered)
            
            # Vérifier la qualité de la réduction
            reconstruction_error = calculate_reconstruction_error(
                X_train_filtered,
                pca.inverse_transform(X_reduced)
            )
            
            if reconstruction_error > 0.1 and recursion_count < max_recursions:
                print("\n⚠️ Erreur de reconstruction trop élevée")
                print("→ Tentative avec KernelPCA")
                return optimize_dimensionality_reduction(
                    X_train_filtered, 
                    method='kpca',
                    min_explained_variance=min_explained_variance,
                    recursion_count=recursion_count + 1
                )
            
            return X_reduced, pca
            
        elif method == 'kpca':
            # Configuration de KernelPCA
            n_components = min(int(X_train_filtered.shape[1] * 0.7), 100)
            kernels = ['rbf', 'poly', 'cosine']
            best_config = None
            best_score = -np.inf
            
            print("\nRecherche de la meilleure configuration KernelPCA:")
            for kernel in kernels:
                for gamma in [0.01, 0.1, 1.0]:
                    try:
                        kpca = KernelPCA(
                            n_components=n_components,
                            kernel=kernel,
                            gamma=gamma,
                            n_jobs=-1
                        )
                        X_kpca = kpca.fit_transform(X_train_filtered)
                        
                        # Évaluer la qualité
                        reconstruction = kpca.inverse_transform(X_kpca)
                        error = calculate_reconstruction_error(X_train_filtered, reconstruction)
                        variances = np.var(X_kpca, axis=0)
                        significant = np.sum(variances >= 1e-5)
                        
                        # Score composite
                        quality_score = (
                            (1 - error) * 0.4 +  # Faible erreur
                            (significant/n_components) * 0.4 +  # Composantes significatives
                            (np.mean(variances)/np.std(variances)) * 0.2  # Bon ratio signal/bruit
                        )
                        
                        print(f"\nKernel: {kernel}, Gamma: {gamma}")
                        print(f"• Erreur: {error:.4f}")
                        print(f"• Composantes significatives: {significant}")
                        print(f"• Score: {quality_score:.4f}")
                        
                        if quality_score > best_score:
                            best_score = quality_score
                            best_config = (kernel, gamma, X_kpca, kpca)
                            
                    except Exception as e:
                        print(f"⚠️ Échec avec {kernel}, gamma={gamma}: {str(e)}")
                        continue
            
            if best_config is None:
                print("\n⚠️ Échec de KernelPCA")
                print("→ Retour à PCA avec paramètres conservateurs")
                return optimize_dimensionality_reduction(
                    X_train_filtered,
                    method='pca',
                    min_explained_variance=0.95,
                    recursion_count=recursion_count + 1
                )
            
            kernel, gamma, X_reduced, kpca = best_config
            print(f"\n✅ Meilleure configuration:")
            print(f"• Kernel: {kernel}")
            print(f"• Gamma: {gamma}")
            print(f"• Score: {best_score:.4f}")
            
            return X_reduced, kpca
            
    except Exception as e:
        print(f"\n⚠️ Erreur lors de la réduction ({method}): {str(e)}")
        if recursion_count < max_recursions:
            print("→ Tentative avec méthode alternative")
            new_method = 'kpca' if method == 'pca' else 'pca'
            return optimize_dimensionality_reduction(
                X_train_filtered,
                method=new_method,
                min_explained_variance=min_explained_variance,
                recursion_count=recursion_count + 1
            )
        else:
            print("→ Utilisation de PCA basique")
            n_components = min(50, X_train_filtered.shape[1], X_train_filtered.shape[0])
            pca = PCA(n_components=n_components)
            return pca.fit_transform(X_train_filtered), pca

def calculate_reconstruction_error(original, reconstructed):
    """Calcule l'erreur de reconstruction avec epsilon pour éviter la division par zéro"""
    epsilon = 1e-8
    error = np.mean(np.abs((original - reconstructed) / (original + epsilon)))
    return error

def apply_progressive_upsampling(X_dict, timeframe, current_ratio, target_ratio=0.20, max_factor=2.0):
    """Applique un upsampling progressif pour les timeframes longs"""
    print(f"\nUpsampling progressif pour {timeframe}:")
    
    # Calculer le facteur initial (2.0x maximum)
    initial_factor = min(2.0, target_ratio / current_ratio)
    data = X_dict[timeframe]['X']  # Accéder aux données X directement
    
    # Première phase d'upsampling avec contrôle adaptatif du bruit
    n_samples = len(data)
    target_samples = int(n_samples * initial_factor)
    
    # Calculer l'échelle de bruit adaptative basée sur la variance des données
    data_std = np.std(data, axis=0)
    base_noise_scale = {
        '1d': 0.0001,
        '1w': 0.0002,
        '1M': 0.0003
    }.get(timeframe, 0.0001)
    
    # Ajuster l'échelle du bruit en fonction de la variance des données
    noise_scale = base_noise_scale * np.mean(data_std)
    
    print(f"Phase d'upsampling:")
    print(f"• Échantillons initiaux: {n_samples}")
    print(f"• Cible: {target_samples}")
    print(f"• Facteur: {initial_factor:.2f}x")
    print(f"• Échelle de bruit adaptative: {noise_scale:.6f}")
    
    # Générer les nouveaux échantillons avec bruit contrôlé
    if target_samples > n_samples:
        # Utiliser SMOTE pour les premiers échantillons
        try:
            print("\nApplication de SMOTE pour la génération initiale...")
            smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=min(5, n_samples-1),
                random_state=42
            )
            
            # Aplatir les données pour SMOTE
            X_flat = data.reshape(n_samples, -1)
            y_dummy = np.zeros(n_samples)  # Labels fictifs pour SMOTE
            
            X_resampled, _ = smote.fit_resample(X_flat, y_dummy)
            new_samples = X_resampled[n_samples:target_samples].reshape(-1, *data.shape[1:])
            
            # Ajouter un bruit minimal pour éviter les duplicats exacts
            noise = np.random.normal(0, noise_scale/10, size=new_samples.shape)
            new_samples += noise
            
        except Exception as e:
            print(f"⚠️ SMOTE a échoué ({str(e)}), utilisation de l'approche par bruit...")
            # Sélection aléatoire avec bruit
            indices = np.random.choice(n_samples, target_samples - n_samples)
            new_samples = data[indices].copy()
            
            # Appliquer un bruit gaussien avec décroissance
            for i, sample in enumerate(new_samples):
                decay = 1 - (i / len(new_samples))  # Décroissance linéaire du bruit
                noise = np.random.normal(0, noise_scale * decay, size=sample.shape)
                new_samples[i] += noise
        
        # Combiner avec les données originales
        upsampled_data = np.concatenate([data, new_samples])
        
        # Vérifier la qualité des données générées
        original_std = np.std(data, axis=0)
        new_std = np.std(upsampled_data, axis=0)
        std_ratio = np.mean(new_std / original_std)
        
        print("\nVérification de la qualité:")
        print(f"• Ratio de déviation standard: {std_ratio:.3f}")
        if std_ratio > 1.1:
            print("⚠️ Variance légèrement augmentée - Ajustement du bruit")
            # Réduire le bruit si nécessaire
            upsampled_data[n_samples:] = (upsampled_data[n_samples:] + data.mean(axis=0)) / 2
    else:
        upsampled_data = data
    
    final_ratio = len(upsampled_data) / len(X_dict['5m']['X'])
    print(f"\nRésultat final:")
    print(f"• Ratio initial: {current_ratio:.3f}")
    print(f"• Ratio final: {final_ratio:.3f}")
    print(f"• Facteur effectif: {len(upsampled_data)/n_samples:.2f}x")
    
    return upsampled_data

def prepare_data_for_reduction(X_train_dict):
    """Prépare les données pour la réduction de dimensionnalité"""
    # Calculer le nombre total de features
    total_features = sum(np.prod(X_train_dict[tf].shape[1:]) for tf in X_train_dict)
    min_samples = min(X_train_dict[tf].shape[0] for tf in X_train_dict)
    
    print(f"\nPréparation des données:")
    print(f"• Total features: {total_features}")
    print(f"• Échantillons: {min_samples}")
    
    # Créer le tableau combiné
    X_combined = np.zeros((min_samples, total_features))
    current_pos = 0
    
    for tf in X_train_dict:
        # Prendre les min_samples premiers échantillons
        data = X_train_dict[tf][:min_samples]
        n_features = np.prod(data.shape[1:])
        
        # Aplatir et ajouter au tableau combiné
        X_combined[:, current_pos:current_pos+n_features] = data.reshape(min_samples, -1)
        current_pos += n_features
        
        print(f"• {tf}: {n_features} features ajoutées")
    
    return X_combined

def balance_neutral_class(X_train, y_train, target_ratio=0.05):
    """Équilibre la classe neutre pour atteindre le ratio cible"""
    print("\nÉquilibrage de la classe neutre:")
    print("="*50)
    
    # Analyser la distribution initiale
    n_samples = len(y_train)
    n_neutral = np.sum(y_train == 0)
    current_ratio = n_neutral / n_samples
    
    print(f"Distribution initiale:")
    print(f"• Total échantillons: {n_samples}")
    print(f"• Échantillons neutres: {n_neutral}")
    print(f"• Ratio actuel: {current_ratio:.3%}")
    print(f"• Ratio cible: {target_ratio:.3%}")
    
    if current_ratio >= target_ratio:
        print("✅ Classe neutre suffisamment représentée")
        return X_train, y_train
    
    # Calculer le nombre d'échantillons à ajouter
    target_neutral = int(n_samples * target_ratio)
    samples_to_add = target_neutral - n_neutral
    
    print(f"\nGénération de {samples_to_add} nouveaux échantillons neutres")
    
    # Approche par interpolation directe
    neutral_indices = np.where(y_train == 0)[0]
    X_train_new = {tf: X_train[tf].copy() for tf in X_train}
    
    # Vérifier la taille minimale pour chaque timeframe
    min_samples = min(X_train[tf].shape[0] for tf in X_train)
    valid_neutral_indices = neutral_indices[neutral_indices < min_samples]
    
    if len(valid_neutral_indices) < 2:
        print("⚠️ Pas assez d'échantillons neutres valides pour l'interpolation")
        return X_train, y_train
    
    print(f"• Échantillons neutres valides: {len(valid_neutral_indices)}")
    
    # Générer les nouveaux échantillons par lots
    batch_size = 1000
    remaining = samples_to_add
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        
        # Sélectionner des paires d'indices aléatoires
        idx1 = np.random.choice(valid_neutral_indices, current_batch)
        idx2 = np.random.choice(valid_neutral_indices, current_batch)
        
        # Générer des coefficients d'interpolation
        alphas = np.random.uniform(0.2, 0.8, current_batch)
        
        for tf in X_train:
            # Créer le batch de nouveaux échantillons
            new_samples = np.zeros((current_batch,) + X_train[tf].shape[1:])
            
            for i in range(current_batch):
                # Interpolation avec bruit
                new_samples[i] = (
                    alphas[i] * X_train[tf][idx1[i]] + 
                    (1 - alphas[i]) * X_train[tf][idx2[i]] +
                    np.random.normal(0, 0.0001, X_train[tf].shape[1:])
                )
            
            # Ajouter les nouveaux échantillons
            X_train_new[tf] = np.concatenate([X_train_new[tf], new_samples])
        
        # Mettre à jour le compteur
        remaining -= current_batch
        
        # Afficher la progression
        if remaining % 5000 == 0:
            print(f"• Progression: {samples_to_add - remaining}/{samples_to_add}")
    
    # Ajouter les labels
    y_train_new = np.concatenate([y_train, np.zeros(samples_to_add)])
    
    # Vérifier la distribution finale
    final_neutral = np.sum(y_train_new == 0)
    final_ratio = final_neutral / len(y_train_new)
    
    print("\nRésultats finaux:")
    print(f"• Échantillons neutres: {final_neutral}")
    print(f"• Ratio final: {final_ratio:.3%}")
    print(f"• Augmentation: {final_neutral/n_neutral:.1f}x")
    
    return X_train_new, y_train_new

def balance_long_timeframes(sequences_dict, min_ratio=0.01):
    """Équilibre les timeframes longs pour assurer une représentation minimale"""
    print("\nÉquilibrage des timeframes longs:")
    print("="*50)
    
    # Calculer la taille de référence (5m)
    base_size = len(sequences_dict['5m']['X'])
    print(f"Taille de référence (5m): {base_size} échantillons")
    
    # Timeframes à équilibrer
    long_timeframes = ['1d', '1w', '1M']
    balanced_dict = sequences_dict.copy()
    
    for tf in long_timeframes:
        if tf not in sequences_dict:
            continue
            
        current_size = len(sequences_dict[tf]['X'])
        current_ratio = current_size / base_size
        
        print(f"\nTimeframe {tf}:")
        print(f"• Taille actuelle: {current_size}")
        print(f"• Ratio actuel: {current_ratio:.3%}")
        
        if current_ratio < min_ratio:
            target_size = int(base_size * min_ratio)
            samples_to_add = target_size - current_size
            
            print(f"• Cible: {target_size} ({min_ratio:.1%} de 5m)")
            print(f"• Échantillons à ajouter: {samples_to_add}")
            
            try:
                # Approche hybride SMOTE + interpolation
                X = sequences_dict[tf]['X']
                y = sequences_dict[tf]['y']
                
                # Phase 1: SMOTE pour doubler les données
                smote_size = min(current_size * 2, target_size)
                if smote_size > current_size:
                    print("\nPhase 1: SMOTE")
                    smote = SMOTE(
                        sampling_strategy='all',
                        k_neighbors=min(5, current_size-1),
                        random_state=42
                    )
                    
                    X_flat = X.reshape(current_size, -1)
                    X_smote, y_smote = smote.fit_resample(X_flat, y)
                    X_new = X_smote.reshape(-1, *X.shape[1:])
                    
                    print(f"• Générés: {len(X_new) - current_size}")
                else:
                    X_new = X
                    y_smote = y
                
                # Phase 2: Interpolation si nécessaire
                remaining = target_size - len(X_new)
                if remaining > 0:
                    print("\nPhase 2: Interpolation")
                    
                    # Générer par lots
                    batch_size = 1000
                    while remaining > 0:
                        current_batch = min(batch_size, remaining)
                        
                        # Sélectionner des paires aléatoires
                        idx1 = np.random.choice(len(X_new), current_batch)
                        idx2 = np.random.choice(len(X_new), current_batch)
                        weights = np.random.uniform(0.3, 0.7, current_batch)
                        
                        # Interpolation avec bruit adaptatif
                        batch_samples = np.zeros((current_batch,) + X.shape[1:])
                        for i in range(current_batch):
                            sample = (
                                weights[i] * X_new[idx1[i]] +
                                (1 - weights[i]) * X_new[idx2[i]]
                            )
                            
                            # Bruit adaptatif basé sur la variance locale
                            local_std = np.std([X_new[idx1[i]], X_new[idx2[i]]], axis=0)
                            noise_scale = 0.1 * local_std
                            noise = np.random.normal(0, noise_scale)
                            
                            batch_samples[i] = sample + noise
                        
                        X_new = np.concatenate([X_new, batch_samples])
                        y_smote = np.concatenate([y_smote, y[idx1]])
                        remaining -= current_batch
                        
                        print(f"• Progression: {target_size - remaining}/{target_size}")
                
                # Vérifier la qualité
                print("\nVérification de la qualité:")
                orig_std = np.std(X, axis=0)
                new_std = np.std(X_new, axis=0)
                std_ratio = np.mean(new_std / orig_std)
                print(f"• Ratio de déviation standard: {std_ratio:.3f}")
                
                # Ajuster si nécessaire
                if std_ratio > 1.2:
                    print("⚠️ Variance trop élevée - Application de régularisation")
                    X_new[current_size:] = (
                        X_new[current_size:] + 
                        np.mean(X, axis=0)
                    ) / 2
                
                # Mettre à jour le dictionnaire
                balanced_dict[tf] = {
                    'X': X_new,
                    'y': y_smote
                }
                
                final_ratio = len(X_new) / base_size
                print(f"\nRésultat final pour {tf}:")
                print(f"• Taille finale: {len(X_new)}")
                print(f"• Ratio final: {final_ratio:.3%}")
                
            except Exception as e:
                print(f"⚠️ Erreur lors de l'équilibrage de {tf}: {str(e)}")
                print("→ Conservation des données originales")
                continue
    
    return balanced_dict

def test_class_balance():
    """Test l'équilibrage des classes dans le préprocesseur"""
    print("Test de l'équilibrage des classes")
    print("="*50)
    
    # 1. Charger les données
    print("\n1. Chargement des données...")
    loader = HistoricalDataLoader()
    data_dict = loader.load_multi_timeframe_data()
    
    # 2. Créer le préprocesseur
    print("\n2. Initialisation du préprocesseur...")
    preprocessor = MLPreprocessor()
    
    # 3. Préparer les séquences avec équilibrage des classes
    print("\n3. Préparation des séquences...")
    sequences = preprocessor.prepare_sequences(
        data_dict,
        sequence_length=15,
        prediction_horizon=12
    )
    
    # 4. Vérifier la distribution des classes pour chaque timeframe
    print("\n4. Vérification des distributions...")
    for tf in sequences.keys():
        print(f"\nTimeframe: {tf}")
        print("-" * 30)
        
        # Distribution globale
        y = sequences[tf]['y']
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        total = sum(counts)
        
        print("Distribution globale:")
        for label in sorted(distribution.keys()):
            count = distribution[label]
            percentage = (count / total) * 100
            print(f"Classe {label}: {count} ({percentage:.1f}%)")
        
        # Distribution train/test
        for split in ['train', 'test']:
            y_split = sequences[tf][f'y_{split}']
            unique, counts = np.unique(y_split, return_counts=True)
            distribution = dict(zip(unique, counts))
            total = sum(counts)
            
            print(f"\nDistribution {split}:")
            for label in sorted(distribution.keys()):
                count = distribution[label]
                percentage = (count / total) * 100
                print(f"Classe {label}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    try:
        # Tester le preprocessor
        X_train, y_train, X_test, y_test = test_ml_preprocessor()
        
        # Tester la génération de features
        features = test_feature_generation()
        
        # Tester l'équilibrage des classes
        test_class_balance()
        
        print("\n✅ Tests terminés avec succès!")
        
    except Exception as e:
        print(f"\n❌ Tests échoués: {str(e)}")
        raise 