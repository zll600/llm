"""
03 - Text Clustering with Embeddings
=====================================

Automatically group similar texts together without labels.

Real-world use cases:
- Organize customer feedback by topic
- Group news articles by theme
- Find duplicate/near-duplicate content
- Discover hidden patterns in data
- Auto-categorize support tickets

"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict

print("=" * 70)
print("TEXT CLUSTERING WITH EMBEDDINGS")
print("=" * 70)

# ============================================================================
# PART 1: The Data - Customer Reviews
# ============================================================================

# Simulated customer reviews for a product
reviews = [
    # Cluster 1: Shipping/Delivery issues
    "Delivery took way too long, almost 3 weeks",
    "Package arrived late and damaged",
    "Shipping was slow, expected it much sooner",
    "Delayed shipment, not happy with delivery time",
    # Cluster 2: Product Quality (Positive)
    "Excellent quality, exactly as described",
    "Great build quality, very satisfied",
    "High quality product, worth the money",
    "Product quality exceeded my expectations",
    # Cluster 3: Product Quality (Negative)
    "Poor quality, broke after one week",
    "Cheap materials, not worth the price",
    "Quality is terrible, very disappointed",
    "Product feels cheap and flimsy",
    # Cluster 4: Customer Service
    "Customer support was very helpful",
    "Great service, they answered all my questions",
    "Support team resolved my issue quickly",
    "Excellent customer service experience",
    # Cluster 5: Value for Money
    "Overpriced for what you get",
    "Great value, good price for quality",
    "Too expensive, found cheaper alternatives",
    "Best bang for your buck, highly recommend",
]

print(f"\nDataset: {len(reviews)} customer reviews")
print("\nSample reviews:")
for i, review in enumerate(reviews[:3]):
    print(f"  {i + 1}. {review}")

# ============================================================================
# PART 2: Generate Embeddings
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING EMBEDDINGS")
print("=" * 70)

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"\nEmbedding {len(reviews)} reviews...")
embeddings = model.encode(reviews, show_progress_bar=True)
print(f"✓ Done! Shape: {embeddings.shape}")

# ============================================================================
# PART 3: K-Means Clustering
# ============================================================================

print("\n" + "=" * 70)
print("CLUSTERING")
print("=" * 70)

# We know there are ~5 topics, so let's use 5 clusters
n_clusters = 5

print(f"\nRunning K-Means with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)
print("✓ Clustering complete!")

# ============================================================================
# PART 4: Analyze Clusters
# ============================================================================

print("\n" + "=" * 70)
print("CLUSTER ANALYSIS")
print("=" * 70)

# Group reviews by cluster
clusters = defaultdict(list)
for review, label in zip(reviews, cluster_labels):
    clusters[label].append(review)

# Show each cluster
for cluster_id in sorted(clusters.keys()):
    print(f"\n{'─' * 70}")
    print(f"CLUSTER {cluster_id} ({len(clusters[cluster_id])} reviews)")
    print(f"{'─' * 70}")
    for review in clusters[cluster_id]:
        print(f"  • {review}")

# ============================================================================
# PART 5: Find Cluster Themes Automatically
# ============================================================================

print("\n" + "=" * 70)
print("IDENTIFYING CLUSTER THEMES")
print("=" * 70)


def get_cluster_centroid_nearest_text(cluster_id: str):
    """
    Find the review closest to the cluster center.
    This is often representative of the cluster theme.
    """
    cluster_reviews_idx = [
        i for i, label in enumerate(cluster_labels) if label == cluster_id
    ]
    cluster_embeddings = embeddings[cluster_reviews_idx]

    # Get cluster centroid
    centroid = kmeans.cluster_centers_[cluster_id]

    # Find closest review to centroid
    similarities = cosine_similarity([centroid], cluster_embeddings)[0]
    closest_idx = cluster_reviews_idx[np.argmax(similarities)]

    return reviews[closest_idx]


print("\nMost representative review for each cluster:\n")
for cluster_id in sorted(clusters.keys()):
    representative = get_cluster_centroid_nearest_text(cluster_id)
    print(f'Cluster {cluster_id}: "{representative}"')

# ============================================================================
# PART 6: Cluster Quality Metrics
# ============================================================================

print("\n" + "=" * 70)
print("CLUSTER QUALITY")
print("=" * 70)


# Intra-cluster similarity (how similar items within a cluster are)
def avg_intra_cluster_similarity(cluster_id):
    """Average similarity within a cluster (higher is better)"""
    cluster_indices = [
        i for i, label in enumerate(cluster_labels) if label == cluster_id
    ]
    if len(cluster_indices) < 2:
        return 0.0

    cluster_embs = embeddings[cluster_indices]
    sims = cosine_similarity(cluster_embs, cluster_embs)

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
    return sims[mask].mean()


# Inter-cluster distance (how far apart clusters are)
inter_cluster_sims = cosine_similarity(kmeans.cluster_centers_)

print("\nIntra-cluster Similarity (higher = more cohesive):")
for cluster_id in sorted(clusters.keys()):
    sim = avg_intra_cluster_similarity(cluster_id)
    print(f"  Cluster {cluster_id}: {sim:.4f}")

print("\nInter-cluster Similarity Matrix (lower = more distinct):")
print("    ", end="")
for i in range(n_clusters):
    print(f"C{i}    ", end="")
print()
for i in range(n_clusters):
    print(f"C{i}: ", end="")
    for j in range(n_clusters):
        if i == j:
            print("  -   ", end="")  # Diagonal
        else:
            print(f"{inter_cluster_sims[i][j]:.3f} ", end="")
    print()

# ============================================================================
# PART 7: Finding Optimal Number of Clusters
# ============================================================================

print("\n" + "=" * 70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)

# Elbow method: try different numbers of clusters
inertias = []
silhouette_scores = []
K_range = range(2, 10)

print("\nTrying different numbers of clusters...")

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(embeddings)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(embeddings, labels_temp))

print("\nClusters  Inertia  Silhouette")
print("─" * 35)
for k, inertia, sil_score in zip(K_range, inertias, silhouette_scores):
    marker = " ← Good choice!" if sil_score == max(silhouette_scores) else ""
    print(f"{k:^9} {inertia:^8.2f} {sil_score:^10.3f}{marker}")

print("\n Silhouette score: -1 (worst) to +1 (best)")
print("   Higher silhouette = better-defined clusters")

# ============================================================================
# PART 8: Practical Applications
# ============================================================================

print("\n" + "=" * 70)
print("PRACTICAL APPLICATIONS")
print("=" * 70)

print("""
1. CUSTOMER FEEDBACK ANALYSIS
   - Auto-categorize feedback into themes
   - Identify trending issues
   - Prioritize product improvements

2. CONTENT ORGANIZATION
   - Group blog posts by topic
   - Organize documentation
   - Categorize support articles

3. DUPLICATE DETECTION
   - Find near-duplicate content
   - Merge similar tickets/requests
   - Identify plagiarism

4. TOPIC DISCOVERY
   - Find hidden themes in data
   - Explore unstructured datasets
   - Research paper clustering

5. RECOMMENDATION SYSTEMS
   - Group similar items
   - "Customers who liked X also liked..."
   - Content-based filtering
""")

# ============================================================================
# PART 9: Advanced: Hierarchical Clustering
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: HIERARCHICAL CLUSTERING")
print("=" * 70)


# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
hier_labels = hierarchical.fit_predict(embeddings)

# Compare with K-means
print("\nComparison: K-Means vs Hierarchical")
print("─" * 70)
print(f"{'Review':<45} K-Means  Hierarchical")
print("─" * 70)
for i, review in enumerate(reviews[:10]):  # Show first 10
    print(f"{review[:45]:<45} {cluster_labels[i]:^7} {hier_labels[i]:^12}")

print("\n Different algorithms may find different patterns!")

# ============================================================================
# PART 10: Your Turn
# ============================================================================

print("\n" + "=" * 70)
print("BUILD YOUR OWN CLUSTERING")
print("=" * 70)

print("""
Try clustering your own data!

Examples:

1. NEWS ARTICLES
   articles = ["...", "...", "..."]
   embeddings = model.encode(articles)
   clusters = KMeans(n_clusters=5).fit_predict(embeddings)

2. SOCIAL MEDIA POSTS
   posts = [tweets, comments, etc.]
   # Discover trending topics automatically

3. PRODUCT DESCRIPTIONS
   products = [description1, description2, ...]
   # Group similar products

4. CODE SNIPPETS
   code_docs = [docstring1, docstring2, ...]
   # Organize code by functionality

5. RESEARCH PAPER ABSTRACTS
   abstracts = [abstract1, abstract2, ...]
   # Find related research

Experiment:
- Try different numbers of clusters
- Use different embedding models
- Try DBSCAN for density-based clustering
- Add dimensionality reduction (PCA) before clustering
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
 Clustering groups similar texts without labels

 Process:
  1. Generate embeddings for all texts
  2. Apply clustering algorithm (K-Means, Hierarchical, DBSCAN)
  3. Analyze clusters to understand themes
  4. Use metrics to evaluate quality

 Choosing number of clusters:
  - Domain knowledge (if you know topics)
  - Elbow method (plot inertia)
  - Silhouette score (cluster quality)

 Applications:
  - Organize large text collections
  - Discover hidden themes
  - Detect duplicates
  - Categorize content automatically

 Next steps:
  - Run 04-similarity.py for analogies and relationships
  - Run 06-visualization.py to SEE your clusters in 2D
  - Combine with 02-semantic-search.py for cluster-based search
""")

print("\n" + "=" * 70)
print("✓ Clustering complete! Try 04-similarity.py next")
print("=" * 70)
