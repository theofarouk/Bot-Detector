"""
Analyse des types d'interactions disponibles dans tweets.csv pour construire le graphe
"""
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

def analyze_tweet_interactions(tweets_file, max_tweets=10000):
    """Analyse les interactions dans un fichier tweets.csv"""
    stats = {
        'total_tweets': 0,
        'replies': 0,
        'retweets': 0,
        'mentions': 0,
        'unique_mentions': set(),
        'reply_targets': set(),
        'retweet_sources': set(),
        'users_with_interactions': set()
    }
    
    print(f"\n📊 Analyse de {tweets_file}")
    
    try:
        # Essayer différents encodings
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                with open(tweets_file, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    
                    for i, tweet in enumerate(reader):
                        if i >= max_tweets:
                            print(f"   (Limite atteinte: {max_tweets} tweets analysés)")
                            break
                            
                        stats['total_tweets'] += 1
                        user_id = tweet.get('user_id')
                        
                        # Analyser les réponses
                        reply_to_user = tweet.get('in_reply_to_user_id', '0')
                        if reply_to_user and reply_to_user != '0':
                            stats['replies'] += 1
                            stats['reply_targets'].add(reply_to_user)
                            stats['users_with_interactions'].add(user_id)
                        
                        # Analyser les retweets  
                        retweeted_id = tweet.get('retweeted_status_id', '0')
                        if retweeted_id and retweeted_id != '0':
                            stats['retweets'] += 1
                            stats['retweet_sources'].add(retweeted_id)
                            stats['users_with_interactions'].add(user_id)
                        
                        # Extraire les mentions depuis le texte
                        text = tweet.get('text', '')
                        mentions = re.findall(r'@(\w+)', text)
                        if mentions:
                            stats['mentions'] += len(mentions)
                            stats['unique_mentions'].update(mentions)
                            stats['users_with_interactions'].add(user_id)
                
                # Si on arrive ici, l'encoding a marché
                print(f"   ✓ Encoding utilisé: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"   ❌ Aucun encoding ne fonctionne")
            return None
                    
        # Convertir sets en counts pour l'affichage
        stats['unique_mentions'] = len(stats['unique_mentions'])
        stats['reply_targets'] = len(stats['reply_targets'])
        stats['retweet_sources'] = len(stats['retweet_sources'])
        stats['users_with_interactions'] = len(stats['users_with_interactions'])
                    
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None
    
    return stats

def print_interaction_summary(stats, dataset_name):
    """Affiche un résumé des interactions"""
    if not stats:
        return
        
    total = stats['total_tweets']
    
    print(f"""
   Tweets analysés        : {total:7,}
   Réponses              : {stats['replies']:7,} ({stats['replies']/total*100:5.1f}%)
   Retweets              : {stats['retweets']:7,} ({stats['retweets']/total*100:5.1f}%)
   Mentions totales      : {stats['mentions']:7,} ({stats['mentions']/total*100:5.1f}%)
   
   Utilisateurs uniques mentionnés : {stats['unique_mentions']:7,}
   Utilisateurs uniques en reply    : {stats['reply_targets']:7,}
   Tweets uniques retweetés         : {stats['retweet_sources']:7,}
   
   🔗 Utilisateurs avec interactions : {stats['users_with_interactions']:7,}
   """)

def main():
    """Analyse les interactions de tous les datasets"""
    base_path = Path("datasets_full.csv")
    
    datasets = [
        ("👥 Genuine Accounts", "genuine_accounts.csv"),
        ("🤖 Social Spambots 1", "social_spambots_1.csv"),
        ("🤖 Social Spambots 2", "social_spambots_2.csv"),
        ("🤖 Fake Followers", "fake_followers.csv"),
    ]
    
    print("="*80)
    print("🔍 ANALYSE DES INTERACTIONS POUR CONSTRUCTION DU GRAPHE")
    print("="*80)
    
    all_stats = {}
    
    for name, folder in datasets:
        tweets_file = base_path / folder / "tweets.csv"
        if tweets_file.exists():
            stats = analyze_tweet_interactions(tweets_file)
            if stats:
                all_stats[name] = stats
                print_interaction_summary(stats, name)
        else:
            print(f"\n❌ {name}: tweets.csv non trouvé")
    
    # Résumé global
    print("="*80)
    print("📈 RÉSUMÉ POUR CONSTRUCTION DU GRAPHE")
    print("="*80)
    
    total_interactions = 0
    total_users_with_interactions = 0
    
    for name, stats in all_stats.items():
        interactions = stats['replies'] + stats['retweets'] + stats['mentions']
        total_interactions += interactions
        total_users_with_interactions += stats['users_with_interactions']
        
        density = stats['users_with_interactions'] / max(stats['total_tweets'], 1)
        print(f"{name:<25} : {interactions:6,} interactions ({density:.1%} utilisateurs actifs)")
    
    print(f"\n🎯 TOTAL INTERACTIONS : {total_interactions:,}")
    print(f"👥 TOTAL UTILISATEURS AVEC INTERACTIONS : {total_users_with_interactions:,}")
    
    # Recommandations
    print("\n" + "="*80)
    print("💡 RECOMMANDATIONS POUR LE GRAPHE")
    print("="*80)
    
    if total_interactions > 10000:
        print("""
✅ EXCELLENT ! Suffisamment d'interactions pour un graphe riche.

Stratégie recommandée :
1. Graphe principal : Interactions réelles (mentions + replies + retweets)
2. Seuillage : Garder les utilisateurs avec au moins N interactions
3. Validation : Vérifier la connectivité du graphe final
        """)
    elif total_interactions > 1000:
        print("""
🔶 MODÉRÉ. Interactions limitées mais utilisables.

Stratégie recommandée :
1. Graphe hybride : Interactions réelles + similarité des features
2. Pondération : Plus de poids aux interactions réelles
3. Fallback : k-NN sur features si composantes déconnectées
        """)
    else:
        print("""
⚠️  FAIBLE. Peu d'interactions disponibles.

Stratégie recommandée :
1. Graphe synthétique : Basé sur similarité des features (cosine similarity)
2. Homophilie simulée : Bots connectés préférentiellement entre eux
3. Validation : Tests avec différents seuils de similarité
        """)

if __name__ == "__main__":
    main()