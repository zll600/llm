"""
01 - Vector Embeddings Basics
==============================

Learn the fundamentals by running code.

What you'll learn:
- What embeddings actually are (spoiler: just arrays of numbers)
- How to generate them
- How to measure similarity
- Why they're useful

Run this file and read the output + comments to understand.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("=" * 70)
print("VECTOR EMBEDDINGS BASICS")
print("=" * 70)

# ============================================================================
# PART 1: What is an embedding?
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: What is an Embedding?")
print("=" * 70)

# Load a pre-trained model
# This model has learned to convert text into 384-dimensional vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed a simple sentence
text = "The cat sat on the mat"
embedding = model.encode(text)

print("\nOriginal text:")
print(f"  '{text}'")
print("\nEmbedding (vector representation):")
print(f"  Shape: {embedding.shape}")
print(f"  Type: {type(embedding)}")
print(f"  First 10 values: {embedding[:10]}")
print(f"\n💡 Key insight: Text is converted to {len(embedding)} numbers!")
print("   These numbers capture the 'meaning' of the text.")

# ============================================================================
# PART 2: Similarity - The Core Concept
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: Measuring Similarity")
print("=" * 70)

# Embed multiple sentences
sentences = [
    "The cat sat on the mat",  # 0
    "The dog sat on the rug",  # 1
    "A feline rested on the carpet",  # 2
    "Python is a programming language",  # 3
    "I love coding in Python",  # 4
]

print(f"\nEmbedding {len(sentences)} sentences...")
embeddings = model.encode(sentences)
print(f"✓ Got embeddings with shape: {embeddings.shape}")
print(f"  ({embeddings.shape[0]} sentences × {embeddings.shape[1]} dimensions)")

# Calculate similarities
print(f"\n{'Sentence Pair':<50} Similarity")
print("-" * 70)

# Similar sentences (about cats)
sim_cat_feline = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
print(f"[0] cat sat... ↔ [2] feline rested...{' ' * 11} {sim_cat_feline:.4f} High!")

# Related sentences (both animals)
sim_cat_dog = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"[0] cat sat... ↔ [1] dog sat...{' ' * 17} {sim_cat_dog:.4f} Medium")

# Unrelated sentences
sim_cat_python = cosine_similarity([embeddings[0]], [embeddings[3]])[0][0]
print(f"[0] cat sat... ↔ [3] Python language...{' ' * 10} {sim_cat_python:.4f} Low")

# Related by topic (both about programming)
sim_python_coding = cosine_similarity([embeddings[3]], [embeddings[4]])[0][0]
print(f"[3] Python... ↔ [4] coding Python...{' ' * 12} {sim_python_coding:.4f} High!")

print("\n💡 Key insight: Similar meanings = higher similarity scores!")
print("   Range: 0.0 (completely different) to 1.0 (identical)")

# ============================================================================
# PART 3: Why Cosine Similarity?
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: Understanding Cosine Similarity")
print("=" * 70)

# Let's look at the raw numbers
v1 = embeddings[0]  # cat sentence
v2 = embeddings[2]  # feline sentence (similar meaning)
v3 = embeddings[3]  # python sentence (different meaning)

# Euclidean distance (not commonly used for embeddings)
euclidean_similar = np.linalg.norm(v1 - v2)
euclidean_different = np.linalg.norm(v1 - v3)

print("\nComparing 'cat' sentence to others:")
print("\nUsing Euclidean Distance:")
print(f"  Similar meaning (cat ↔ feline):    {euclidean_similar:.4f}")
print(f"  Different meaning (cat ↔ python):  {euclidean_different:.4f}")
print("  → Lower is more similar")

print("\nUsing Cosine Similarity:")
print(f"  Similar meaning (cat ↔ feline):    {sim_cat_feline:.4f}")
print(f"  Different meaning (cat ↔ python):  {sim_cat_python:.4f}")
print("  → Higher is more similar")

print("\n💡 Key insight: Cosine measures the ANGLE between vectors")
print("   It ignores magnitude, focusing on direction = meaning")

# ============================================================================
# PART 4: Practical Application - Find Most Similar
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: Practical Use - Finding Similar Text")
print("=" * 70)

# A collection of movie descriptions
movies = [
    "A group of astronauts travel through a wormhole in space",
    "A thief who steals corporate secrets through dream infiltration",
    "Two imprisoned men bond over years, finding redemption",
    "A computer hacker learns reality is a simulation",
    "A young wizard attends a magical school",
    "Dinosaurs are brought back to life in a theme park",
    "A hobbit must destroy a powerful ring",
    "A robot left on Earth falls in love",
    "Superheroes team up to save the world",
    "A chef rat controls a human to cook in Paris",
]

# Embed all movies
print(f"Indexing {len(movies)} movie descriptions...")
movie_embeddings = model.encode(movies)
print("✓ Done!")

# Search query
query = "space adventure with sci-fi elements"
print(f"\nQuery: '{query}'")
query_embedding = model.encode(query)

# Find most similar
similarities = cosine_similarity([query_embedding], movie_embeddings)[0]

# Sort by similarity
ranked_indices = np.argsort(similarities)[::-1]  # Descending order

print("\nTop 5 most similar movies:")
print("-" * 70)
for i, idx in enumerate(ranked_indices[:5], 1):
    print(f"{i}. [{similarities[idx]:.4f}] {movies[idx]}")

print("\n💡 Key insight: This is SEMANTIC search!")
print("   We found 'space' and 'sci-fi' movies even though")
print("   those exact words might not appear in the descriptions.")

# ============================================================================
# PART 5: Experiment Yourself
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: Your Turn to Experiment!")
print("=" * 70)

print("""
Try these modifications:

1. Change the query above to find different movies
   - "magical fantasy world"
   - "redemption and prison"
   - "artificial intelligence"

2. Add your own movie descriptions to the list

3. Try different text types:
   - Product descriptions
   - News headlines
   - Code documentation
   - Customer reviews

4. Embed single words:
   - Do "happy" and "joyful" have high similarity?
   - What about "king" and "queen"?

5. Compare different models:
   - all-MiniLM-L6-v2 (small, fast)
   - all-mpnet-base-v2 (larger, more accurate)
   - paraphrase-multilingual-MiniLM-L12-v2 (multi-language)

Just modify the code above and run again!
""")

# ============================================================================
# PART 6: Key Takeaways
# ============================================================================
print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
Embeddings are vectors (arrays of numbers) that represent text

Similar meanings → vectors that are close together

We measure closeness with cosine similarity (0 to 1)

Pre-trained models (like sentence-transformers) do the hard work

Applications:
   - Semantic search (find by meaning, not keywords)
   - Recommendation (find similar items)
   - Clustering (group related texts)
   - Classification (with better features)
   - Duplicate detection
   - And much more!

Next steps:
   - Run 02-semantic-search.py to build a real search engine
   - Run 04-similarity.py to explore analogies (king - man + woman = queen)
   - Run 06-visualization.py to SEE embeddings in 2D/3D
""")

print("\n" + "=" * 70)
print("✓ Basics complete! Ready for 02-semantic-search.py")
print("=" * 70)
