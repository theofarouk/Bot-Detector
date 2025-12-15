## Bot detector in social network by GNNs

Lien vers le dataset de données d'entraînement/test Cresci-2017 :
 - https://service.tib.eu/ldmservice/dataset/cresci2017
 - https://botometer.osome.iu.edu/bot-repository/datasets.html


Ce mini-projet s'inscrit dans le prolongement du TP sur les Graph Convolutional Networks (GCN) réalisé en cours avec Omar Ikne. L'objectif principal était d'étendre les concepts théoriques vers une application concrète : la détection automatique de comptes frauduleux/fermes à troll dans les réseaux sociaux.

Le projet compare les performances des Graph Convolutional Networks (GCN) et Graph Attention Networks (GAT) en utilisant le dataset Cresci-2017 (4,465 comptes Twitter). Face à l'absence de graphe social explicite, nous avons développé une approche innovante de construction de graphe k-NN basée sur la similarité comportementale entre utilisateurs (20 features extraites). L'approche k-NN calcule la similarité cosine entre les features comportementales de chaque compte et connecte chaque utilisateur à ses k=20 voisins les plus similaires, créant ainsi un graphe synthétique avec 95.1% d'homophilie.

L'architecture GAT exploitée introduit un mécanisme d'attention qui apprend automatiquement l'importance relative de chaque voisin lors de l'agrégation des informations. Contrairement au GCN qui applique une moyenne uniforme, le GAT calcule des poids d'attention α_ij pour chaque arête et utilise 4 têtes d'attention multiples pour capturer différents types de patterns relationnels simultanément.

En résulats, nous avons eu : GAT 97.9%, GCN 97.3%. Ce faible écart s'explique par la qualité exceptionnelle des features comportementales et la construction d'un graphe k-NN "trop parfait" où tous les voisins sont déjà optimalement choisis, rendant les mécanismes d'attention superflus. Le projet démontre ainsi l'importance cruciale de valider l'apport réel des architectures complexes, ici il manquait du bruit dans nos données pour démontrer une efficacité plus net du GAT. 