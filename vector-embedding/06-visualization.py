"""
06 - Visualizing Embeddings
============================

See high-dimensional embeddings in 2D/3D to build intuition.

What you'll learn:
- How to reduce 384 dimensions to 2D
- PCA vs t-SNE vs UMAP
- Visualize clusters and relationships
- Debug embedding quality

This helps you understand what's happening "under the hood"!
"""

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("VISUALIZING EMBEDDINGS")
print("=" * 70)

# ============================================================================
# PART 1: Create a Dataset with Clear Groups
# ============================================================================

# Create texts with obvious semantic groups
texts = [
    # Group 1: Animals
    "The cat sat on the mat",
    "A dog barks loudly",
    "The bird flies high",
    "Elephants have long trunks",
    "Fish swim in the ocean",
    # Group 2: Technology
    "Python is a programming language",
    "Machine learning uses neural networks",
    "JavaScript runs in web browsers",
    "Databases store structured data",
    "APIs enable software communication",
    # Group 3: Food
    "Pizza is an Italian dish",
    "Sushi is made with rice and fish",
    "Tacos are a Mexican food",
    "Pasta comes in many shapes",
    "Burgers are popular fast food",
    # Group 4: Sports
    "Soccer is played with a ball",
    "Basketball requires a hoop",
    "Tennis uses a racket and net",
    "Swimming is an Olympic sport",
    "Baseball has nine innings",
    # Group 5: Weather
    "Rain falls from clouds",
    "Snow is frozen precipitation",
    "Sunshine brings warm weather",
    "Hurricanes are powerful storms",
    "Lightning occurs during thunderstorms",
]

# Labels for coloring
labels = (
    ["Animal"] * 5
    + ["Technology"] * 5
    + ["Food"] * 5
    + ["Sports"] * 5
    + ["Weather"] * 5
)

# Color mapping
label_to_color = {
    "Animal": "red",
    "Technology": "blue",
    "Food": "green",
    "Sports": "orange",
    "Weather": "purple",
}

colors = [label_to_color[label] for label in labels]

print(f"\nDataset: {len(texts)} sentences across 5 categories")
print("Sample:")
for i in range(0, 15, 5):
    print(f"  [{labels[i]}] {texts[i]}")

# ============================================================================
# PART 2: Generate Embeddings
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING EMBEDDINGS")
print("=" * 70)

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"\nEmbedding {len(texts)} sentences...")
embeddings = model.encode(texts, show_progress_bar=True)
print(f"✓ Embeddings shape: {embeddings.shape}")
print("  We have 384-dimensional vectors... too many to visualize!")

# ============================================================================
# PART 3: Dimensionality Reduction with PCA
# ============================================================================

print("\n" + "=" * 70)
print("DIMENSIONALITY REDUCTION - PCA")
print("=" * 70)

# Reduce to 2D using PCA
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

print(f"\nReduced from {embeddings.shape[1]}D to 2D using PCA")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print("  (How much information is preserved)")

# Plot
plt.figure(figsize=(12, 8))

for label in set(labels):
    # Get indices for this label
    indices = [i for i, l in enumerate(labels) if l == label]
    x = embeddings_2d_pca[indices, 0]
    y = embeddings_2d_pca[indices, 1]

    plt.scatter(x, y, c=label_to_color[label], label=label, s=100, alpha=0.7)

    # Add text labels
    for idx in indices:
        plt.annotate(
            texts[idx][:15] + "...",
            (embeddings_2d_pca[idx, 0], embeddings_2d_pca[idx, 1]),
            fontsize=8,
            alpha=0.7,
        )

plt.title("Embeddings Visualization using PCA", fontsize=16)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig("vector-embedding/embeddings_pca.png", dpi=150, bbox_inches="tight")
print("\n✓ Plot saved to 'vector-embedding/embeddings_pca.png'")
print("  Open the image to see the visualization!")

# ============================================================================
# PART 4: t-SNE for Better Separation
# ============================================================================

print("\n" + "=" * 70)
print("DIMENSIONALITY REDUCTION - t-SNE")
print("=" * 70)

print("\nRunning t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings_2d_tsne = tsne.fit_transform(embeddings)
print("✓ Done!")

# Plot
plt.figure(figsize=(12, 8))

for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    x = embeddings_2d_tsne[indices, 0]
    y = embeddings_2d_tsne[indices, 1]

    plt.scatter(x, y, c=label_to_color[label], label=label, s=100, alpha=0.7)

    # Add text labels
    for idx in indices:
        plt.annotate(
            texts[idx][:15] + "...",
            (embeddings_2d_tsne[idx, 0], embeddings_2d_tsne[idx, 1]),
            fontsize=8,
            alpha=0.7,
        )

plt.title("Embeddings Visualization using t-SNE", fontsize=16)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("vector-embedding/embeddings_tsne.png", dpi=150, bbox_inches="tight")
print("✓ Plot saved to 'vector-embedding/embeddings_tsne.png'")

# ============================================================================
# PART 5: Compare PCA vs t-SNE
# ============================================================================

print("\n" + "=" * 70)
print("PCA vs t-SNE COMPARISON")
print("=" * 70)

print("""
PCA (Principal Component Analysis):
  ✅ Fast
  ✅ Deterministic (same result every time)
  ✅ Linear method
  ✅ Good for understanding overall structure
  ❌ May not separate clusters well
  ❌ Assumes linear relationships

t-SNE (t-Distributed Stochastic Neighbor Embedding):
  ✅ Better at separating clusters
  ✅ Preserves local structure
  ✅ Non-linear method
  ❌ Slower (especially for large datasets)
  ❌ Non-deterministic (different each run)
  ❌ Can be misleading (artifacts)

Rule of thumb:
- Use PCA for quick exploration
- Use t-SNE for final visualization
- Use UMAP for very large datasets
""")

# ============================================================================
# PART 6: 3D Visualization
# ============================================================================

print("\n" + "=" * 70)
print("3D VISUALIZATION")
print("=" * 70)

# Reduce to 3D
pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(embeddings)

print("\nReduced to 3D")
print(f"Explained variance: {pca_3d.explained_variance_ratio_.sum():.2%}")

# 3D plot

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    x = embeddings_3d[indices, 0]
    y = embeddings_3d[indices, 1]
    z = embeddings_3d[indices, 2]

    ax.scatter(x, y, z, c=label_to_color[label], label=label, s=100, alpha=0.7)

ax.set_title("3D Embedding Visualization (PCA)", fontsize=16)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()

plt.savefig("vector-embedding/embeddings_3d.png", dpi=150, bbox_inches="tight")
print("✓ 3D plot saved to 'vector-embedding/embeddings_3d.png'")

# ============================================================================
# PART 7: Analyzing Embedding Space
# ============================================================================

print("\n" + "=" * 70)
print("ANALYZING THE EMBEDDING SPACE")
print("=" * 70)

from sklearn.metrics.pairwise import cosine_similarity

# Intra-cluster similarity
print("\nAverage similarity within each category:")
for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    group_embeddings = embeddings[indices]

    # Calculate pairwise similarities
    sims = cosine_similarity(group_embeddings)

    # Get upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
    avg_sim = sims[mask].mean()

    print(f"  {label:12}: {avg_sim:.4f}")

# Inter-cluster similarity
print("\nSimilarity between categories (should be lower):")
print(f"{'':12}", end="")
for label in sorted(set(labels)):
    print(f"{label[:8]:>10}", end="")
print()

for label1 in sorted(set(labels)):
    print(f"{label1:12}", end="")
    indices1 = [i for i, l in enumerate(labels) if l == label1]
    emb1 = embeddings[indices1].mean(axis=0, keepdims=True)

    for label2 in sorted(set(labels)):
        indices2 = [i for i, l in enumerate(labels) if l == label2]
        emb2 = embeddings[indices2].mean(axis=0, keepdims=True)

        sim = cosine_similarity(emb1, emb2)[0][0]
        print(f"{sim:>10.4f}", end="")
    print()

print("\n💡 Higher similarity within categories = good clustering!")
print("   Lower similarity between categories = good separation!")

# ============================================================================
# PART 8: Visualize Specific Relationships
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZING SPECIFIC RELATIONSHIPS")
print("=" * 70)

# Create analogies
word_pairs = [
    ("The cat sat on the mat", "A dog barks loudly"),  # Animals
    ("Python is a programming language", "JavaScript runs in web browsers"),  # Tech
    ("Pizza is an Italian dish", "Sushi is made with rice and fish"),  # Food
]

print("\nVisualizing similar pairs:")
plt.figure(figsize=(12, 8))

for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    x = embeddings_2d_tsne[indices, 0]
    y = embeddings_2d_tsne[indices, 1]
    plt.scatter(x, y, c=label_to_color[label], label=label, s=100, alpha=0.3)

# Highlight specific pairs with arrows
for text1, text2 in word_pairs:
    idx1 = texts.index(text1)
    idx2 = texts.index(text2)

    x1, y1 = embeddings_2d_tsne[idx1]
    x2, y2 = embeddings_2d_tsne[idx2]

    plt.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2, color="black", alpha=0.6),
    )

    plt.scatter(
        [x1, x2],
        [y1, y2],
        c="black",
        s=200,
        alpha=0.8,
        edgecolors="yellow",
        linewidths=2,
    )

plt.title("Visualizing Similar Pairs", fontsize=16)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "vector-embedding/embeddings_relationships.png", dpi=150, bbox_inches="tight"
)
print("✓ Saved to 'vector-embedding/embeddings_relationships.png'")

# ============================================================================
# PART 9: Practical Tips
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION TIPS")
print("=" * 70)

print("""
1. CHOOSING THE RIGHT METHOD
   - PCA: Quick exploration, large datasets
   - t-SNE: Better separation, final viz
   - UMAP: Best of both worlds (if available)

2. TUNING t-SNE
   - perplexity: Balance local vs global structure (5-50)
   - n_iter: More iterations = better convergence (1000-5000)
   - random_state: Set for reproducibility

   Example:
   tsne = TSNE(n_components=2,
               perplexity=30,
               n_iter=3000,
               random_state=42)

3. DEALING WITH LARGE DATASETS
   - Use PCA first to reduce to ~50D
   - Then apply t-SNE to the PCA output
   - Or use UMAP (faster than t-SNE)

   Example:
   pca = PCA(n_components=50)
   embeddings_50d = pca.fit_transform(embeddings)
   tsne = TSNE(n_components=2)
   embeddings_2d = tsne.fit_transform(embeddings_50d)

4. INTERACTIVE VISUALIZATION
   - Use plotly for interactive plots
   - Hover to see text
   - Zoom and pan

   Example:
   import plotly.express as px
   import pandas as pd

   df = pd.DataFrame({
       'x': embeddings_2d[:, 0],
       'y': embeddings_2d[:, 1],
       'text': texts,
       'category': labels
   })

   fig = px.scatter(df, x='x', y='y', color='category',
                   hover_data=['text'])
   fig.show()

5. DEBUGGING WITH VISUALIZATION
   - Check if similar items cluster together
   - Find outliers
   - Identify mislabeled data
   - Understand model behavior
""")

# ============================================================================
# PART 10: Your Turn
# ============================================================================

print("\n" + "=" * 70)
print("TRY IT YOURSELF")
print("=" * 70)

print("""
Experiments to try:

1. ADD YOUR OWN DATA
   - Replace `texts` with your own sentences
   - See how they cluster

2. COMPARE MODELS
   - Try different embedding models
   - Visualize and compare

3. FIND OUTLIERS
   - Which points don't fit their cluster?
   - Why might that be?

4. WATCH LEARNING PROGRESS
   - If training embeddings, visualize at each epoch
   - See how clusters form over time

5. MULTILINGUAL
   - Mix English, Spanish, Chinese sentences on same topics
   - Do they cluster by topic or language?
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
✅ Embeddings are high-dimensional (384D, 768D, etc.)

✅ We reduce dimensions to visualize:
   - PCA: Fast, linear, deterministic
   - t-SNE: Better clusters, slower, non-linear
   - UMAP: Best of both (requires install)

✅ Visualization helps:
   - Understand embedding quality
   - Debug clustering
   - Find outliers
   - Build intuition

✅ Good embeddings show:
   - Similar items cluster together
   - Clear separation between different topics
   - Meaningful geometric relationships

✅ Check the saved images:
   - embeddings_pca.png
   - embeddings_tsne.png
   - embeddings_3d.png
   - embeddings_relationships.png

✅ Next steps:
   - Run 04-similarity.py for vector arithmetic
   - Run 07-contextual.py to see how context changes embeddings
   - Visualize your own data!
""")

print("\n" + "=" * 70)
print("✓ Visualization complete! Check the generated images!")
print("=" * 70)
