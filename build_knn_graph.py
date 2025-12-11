"""
Construction d'un graphe k-NN intelligent pour la detection de bots
Base sur des features comportementales plutot que des interactions reelles
"""
import csv
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from datetime import datetime

class BehavioralKNNGraph:
    def __init__(self):
        self.users = {}
        self.user_id_to_index = {}
        self.index_to_user_id = {}
        self.edges = []
        
    def load_users(self, datasets: List[Tuple[str, str, int]]):
        """Charge les utilisateurs depuis les fichiers users.csv"""
        print("Chargement des utilisateurs...")
        
        base_path = Path("datasets_full.csv")
        
        for dataset_name, folder_name, label in datasets:
            users_file = base_path / folder_name / "users.csv"
            
            if not users_file.exists():
                print(f"   ERREUR {dataset_name}: users.csv non trouve")
                continue
            
            count = 0
            try:
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        with open(users_file, 'r', encoding=encoding) as f:
                            reader = csv.DictReader(f)
                            
                            for user_data in reader:
                                user_id = user_data['id']
                                
                                # Nettoyer les donnees
                                for key, value in user_data.items():
                                    if value == '' or value is None:
                                        user_data[key] = '0'
                                
                                user_data['label'] = label
                                user_data['dataset'] = dataset_name
                                
                                self.users[user_id] = user_data
                                count += 1
                        
                        print(f"   OK {dataset_name}: {count:,} utilisateurs (label={label})")
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"   ERREUR {dataset_name}: {e}")
        
        # Creer l'index des noeuds
        for i, user_id in enumerate(self.users.keys()):
            self.user_id_to_index[user_id] = i
            self.index_to_user_id[i] = user_id
        
        print(f"\nTotal: {len(self.users):,} utilisateurs charges")
        
        # Statistiques
        labels = [user['label'] for user in self.users.values()]
        label_counts = Counter(labels)
        print(f"   - Humains (0): {label_counts[0]:,}")
        print(f"   - Bots (1): {label_counts[1]:,}")

    def extract_behavioral_features(self):
        """Extrait des features comportementales avancees"""
        print("\nExtraction des features comportementales...")
        
        behavioral_features = []
        feature_names = []
        
        for user_id in self.index_to_user_id.values():
            user_data = self.users[user_id]
            features = []
            
            # === FEATURES NUMERIQUES DE BASE ===
            followers = float(user_data.get('followers_count', 0))
            friends = float(user_data.get('friends_count', 0))
            statuses = float(user_data.get('statuses_count', 0))
            favourites = float(user_data.get('favourites_count', 0))
            listed = float(user_data.get('listed_count', 0))
            
            # 1. Activite brute (log-transform pour normaliser)
            features.extend([
                np.log1p(followers),
                np.log1p(friends), 
                np.log1p(statuses),
                np.log1p(favourites),
                np.log1p(listed)
            ])
            
            # === FEATURES COMPORTEMENTALES CALCULEES ===
            
            # 2. Ratios sociaux (indicateurs de comportement suspect)
            follower_friend_ratio = followers / (friends + 1)  # Bots ont souvent un ratio faible
            friends_followers_ratio = friends / (followers + 1)  # Inverse
            activity_popularity_ratio = statuses / (followers + 1)  # Beaucoup tweeter, peu de followers
            
            features.extend([
                np.log1p(follower_friend_ratio),
                np.log1p(friends_followers_ratio), 
                np.log1p(activity_popularity_ratio)
            ])
            
            # 3. Indicateurs d'engagement
            if statuses > 0:
                like_per_tweet = favourites / statuses
                list_per_follower = listed / (followers + 1)
            else:
                like_per_tweet = 0
                list_per_follower = 0
                
            features.extend([
                np.log1p(like_per_tweet),
                np.log1p(list_per_follower)
            ])
            
            # 4. Features booleennes (comportement typique des bots)
            default_profile = 1.0 if user_data.get('default_profile', 'False') == 'True' else 0.0
            default_image = 1.0 if user_data.get('default_profile_image', 'False') == 'True' else 0.0
            verified = 1.0 if user_data.get('verified', 'False') == 'True' else 0.0
            geo_enabled = 1.0 if user_data.get('geo_enabled', 'False') == 'True' else 0.0
            protected = 1.0 if user_data.get('protected', 'False') == 'True' else 0.0
            
            features.extend([
                default_profile,
                default_image,
                verified,
                geo_enabled,
                protected
            ])
            
            # 5. Features temporelles (si disponible)
            created_at = user_data.get('created_at', '')
            if created_at:
                try:
                    # Essayer de parser la date pour calculer l'age du compte
                    # Format Twitter: "Mon Apr 26 06:01:55 +0000 2010"
                    created_date = datetime.strptime(created_at.split('+')[0].strip(), "%a %b %d %H:%M:%S")
                    account_age_days = (datetime.now() - created_date.replace(year=2010)).days
                    
                    # Frequence de tweets (activite par jour)
                    tweets_per_day = statuses / max(account_age_days, 1)
                    
                    features.extend([
                        np.log1p(account_age_days),
                        np.log1p(tweets_per_day)
                    ])
                except:
                    # Si parsing echoue, utiliser des valeurs par defaut
                    features.extend([5.0, 1.0])  # ~150 jours, 1 tweet/jour
            else:
                features.extend([5.0, 1.0])
            
            # 6. Features de profil (detection de patterns automatises)
            screen_name = user_data.get('screen_name', '')
            name = user_data.get('name', '')
            description = user_data.get('description', '')
            
            # Presence de chiffres dans le screen_name (typique des bots)
            digits_in_screen_name = sum(c.isdigit() for c in screen_name) / max(len(screen_name), 1)
            
            # Longueur de la description (bots ont souvent des descriptions vides ou generiques)
            description_length = len(description)
            
            # Screen_name == name (peu d'effort de personnalisation)
            name_screen_name_similar = 1.0 if screen_name.lower() == name.lower() else 0.0
            
            features.extend([
                digits_in_screen_name,
                np.log1p(description_length),
                name_screen_name_similar
            ])
            
            behavioral_features.append(features)
        
        # Noms des features pour reference
        if not hasattr(self, 'feature_names'):
            self.feature_names = [
                'followers_log', 'friends_log', 'statuses_log', 'favourites_log', 'listed_log',
                'follower_friend_ratio_log', 'friends_followers_ratio_log', 'activity_popularity_ratio_log',
                'like_per_tweet_log', 'list_per_follower_log',
                'default_profile', 'default_image', 'verified', 'geo_enabled', 'protected',
                'account_age_days_log', 'tweets_per_day_log',
                'digits_in_screen_name', 'description_length_log', 'name_screen_name_similar'
            ]
        
        behavioral_features = np.array(behavioral_features)
        print(f"   OK Features extraites: {behavioral_features.shape}")
        print(f"   Dimensions: {len(self.feature_names)} features par utilisateur")
        
        return behavioral_features
    
    def normalize_features(self, features):
        """Normalise les features pour un meilleur calcul de distance"""
        print("Normalisation des features...")
        
        # Standardisation (mean=0, std=1) pour chaque feature
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[1]):
            feature_col = features[:, i]
            mean_val = np.mean(feature_col)
            std_val = np.std(feature_col)
            
            if std_val > 0:
                normalized[:, i] = (feature_col - mean_val) / std_val
            else:
                normalized[:, i] = feature_col - mean_val
        
        print(f"   OK Features normalisees: mean={np.mean(normalized):.3f}, std={np.std(normalized):.3f}")
        return normalized
    
    def build_knn_graph(self, features, k=20, distance_threshold=None):
        """Construit le graphe k-NN base sur la similarite comportementale"""
        print(f"\nConstruction du graphe k-NN (k={k})...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Utiliser cosine similarity pour capturer les patterns comportementaux
        knn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
        knn.fit(features)
        
        # Trouver les k plus proches voisins pour chaque noeud
        distances, indices = knn.kneighbors(features)
        
        edges_added = 0
        
        for i in range(len(features)):
            source_id = self.index_to_user_id[i]
            
            # Ignorer le premier voisin (c'est le noeud lui-meme)
            for j in range(1, k+1):
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Convertir distance cosinus en similarite (1 - distance)
                similarity = 1.0 - distance
                
                # Optionnel: filtrer par seuil de distance
                if distance_threshold and distance > distance_threshold:
                    continue
                
                target_id = self.index_to_user_id[neighbor_idx]
                
                # Ajouter l'arete (non-dirigee, donc on l'ajoute dans les deux sens)
                # Mais on evite les doublons en verifiant l'ordre
                if i < neighbor_idx:  # Pour eviter les doublons
                    self.edges.append((source_id, target_id, similarity, 'knn_behavioral'))
                    edges_added += 1
        
        print(f"   OK {edges_added:,} aretes ajoutees")
        print(f"   Densite moyenne des connexions: {edges_added*2/len(features):.1f} aretes par noeud")
        
        # Analyser la structure du graphe
        self._analyze_graph_structure(features)
    
    def _analyze_graph_structure(self, features):
        """Analyse la structure du graphe cree"""
        print("\nAnalyse de la structure du graphe:")
        
        # Connectivite par classe
        human_edges = 0
        bot_edges = 0
        cross_edges = 0
        
        for source_id, target_id, weight, edge_type in self.edges:
            source_label = self.users[source_id]['label']
            target_label = self.users[target_id]['label']
            
            if source_label == 0 and target_label == 0:
                human_edges += 1
            elif source_label == 1 and target_label == 1:
                bot_edges += 1
            else:
                cross_edges += 1
        
        total_edges = len(self.edges)
        
        print(f"   - Aretes humain-humain: {human_edges:,} ({human_edges/total_edges*100:.1f}%)")
        print(f"   - Aretes bot-bot: {bot_edges:,} ({bot_edges/total_edges*100:.1f}%)")
        print(f"   - Aretes mixtes: {cross_edges:,} ({cross_edges/total_edges*100:.1f}%)")
        
        # Homophilie (tendance a se connecter a sa propre classe)
        homophily = (human_edges + bot_edges) / total_edges
        print(f"   - Homophilie: {homophily:.3f} (>0.5 = bonne separation)")
        
        if homophily > 0.7:
            print("   EXCELLENT: Le graphe montre une forte homophilie")
        elif homophily > 0.5:
            print("   BON: Le graphe montre une homophilie raisonnable")
        else:
            print("   ATTENTION: Faible homophilie, les classes sont melangees")
    
    def get_adjacency_matrix(self):
        """Retourne la matrice d'adjacence"""
        n_nodes = len(self.users)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for source_id, target_id, weight, edge_type in self.edges:
            source_idx = self.user_id_to_index[source_id]
            target_idx = self.user_id_to_index[target_id]
            adj_matrix[source_idx, target_idx] = weight
            adj_matrix[target_idx, source_idx] = weight  # Graphe non-dirige
        
        return adj_matrix
    
    def get_node_features_and_labels(self, behavioral_features):
        """Retourne les features normalisees et les labels"""
        labels = []
        for user_id in self.index_to_user_id.values():
            labels.append(int(self.users[user_id]['label']))
        
        return behavioral_features, np.array(labels)
    
    def save_graph(self, behavioral_features, output_dir="graph_data"):
        """Sauvegarde le graphe k-NN"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSauvegarde du graphe dans {output_dir}/...")
        
        # Matrice d'adjacence
        adj_matrix = self.get_adjacency_matrix()
        np.save(output_path / "adjacency_matrix.npy", adj_matrix)
        
        # Features et labels
        features, labels = self.get_node_features_and_labels(behavioral_features)
        np.save(output_path / "node_features.npy", features)
        np.save(output_path / "node_labels.npy", labels)
        
        # Metadonnees
        metadata = {
            'graph_type': 'k-NN behavioral',
            'n_nodes': len(self.users),
            'n_edges': len(self.edges),
            'feature_names': self.feature_names,
            'user_id_to_index': self.user_id_to_index,
            'index_to_user_id': {int(k): v for k, v in self.index_to_user_id.items()}
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Statistiques
        stats = {
            'n_nodes': len(self.users),
            'n_edges': len(self.edges),
            'density': len(self.edges) / (len(self.users) * (len(self.users) - 1) / 2),
            'avg_degree': len(self.edges) * 2 / len(self.users),
            'label_distribution': {
                'humans': int(np.sum(labels == 0)),
                'bots': int(np.sum(labels == 1))
            }
        }
        
        with open(output_path / "graph_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"   OK Graphe sauvegarde:")
        print(f"     - Noeuds: {len(self.users):,}")
        print(f"     - Aretes: {len(self.edges):,}")
        print(f"     - Features: {features.shape[1]}")
        print(f"     - Densite: {stats['density']:.6f}")
        print(f"     - Degre moyen: {stats['avg_degree']:.1f}")
        
        return output_path

def main():
    """Construction d'un graphe k-NN comportemental pour detection de bots"""
    
    print("="*80)
    print("CONSTRUCTION GRAPHE k-NN COMPORTEMENTAL")
    print("="*80)
    
    # Datasets a utiliser
    datasets = [
        ("Genuine Accounts", "genuine_accounts.csv", 0),
        ("Social Spambots 1", "social_spambots_1.csv", 1),
        # Optionnel: ajouter plus de donnees
        # ("Social Spambots 2", "social_spambots_2.csv", 1),
    ]
    
    # Construire le graphe
    graph_builder = BehavioralKNNGraph()
    
    # 1. Charger les utilisateurs
    graph_builder.load_users(datasets)
    
    # 2. Extraire les features comportementales avancees
    behavioral_features = graph_builder.extract_behavioral_features()
    
    # 3. Normaliser les features
    normalized_features = graph_builder.normalize_features(behavioral_features)
    
    # 4. Construire le graphe k-NN
    # k=20 donne une bonne connectivite sans trop de bruit
    graph_builder.build_knn_graph(normalized_features, k=20)
    
    # 5. Sauvegarder
    output_path = graph_builder.save_graph(normalized_features)
    
    print("\n" + "="*80)
    print("GRAPHE k-NN COMPORTEMENTAL CONSTRUIT AVEC SUCCES !")
    print("="*80)
    print(f"\nFichiers disponibles dans {output_path}/:")
    print("   - adjacency_matrix.npy : Matrice d'adjacence du graphe")
    print("   - node_features.npy    : Features comportementales normalisees")
    print("   - node_labels.npy      : Labels (0=humain, 1=bot)")
    print("   - metadata.json        : Metadonnees et noms des features")
    print("   - graph_stats.json     : Statistiques du graphe")
    
    print("\nSTRATEGIE:")
    print("- Graphe base sur similarite comportementale (cosine)")
    print("- Chaque noeud connecte a ses 20 voisins les plus similaires")
    print("- Features: ratios sociaux, activite, profil, temporelles")
    print("- Homophilie naturelle: bots similaires connectes entre eux")
    
    print("\nPROCHAINES ETAPES:")
    print("1. Implementer baseline (MLP/Random Forest)")
    print("2. Implementer GCN et GAT") 
    print("3. Comparer les performances")
    print("4. Analyser les poids d'attention (GAT)")

if __name__ == "__main__":
    main()