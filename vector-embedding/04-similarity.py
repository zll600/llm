"""
04 - Similarity & Vector Arithmetic
====================================

Explore relationships in embedding space through vector arithmetic.

Real-world use cases:
- Word analogies (king - man + woman = queen)
- Finding related items
- Recommendation systems
- Understanding semantic relationships
- Query expansion for search

This shows the "magic" of embeddings - relationships become geometric!
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from numpy import ndarray

print("=" * 70)
print("SIMILARITY & VECTOR ARITHMETIC")
print("=" * 70)

# ============================================================================
# PART 1: Load Model and Create Embeddings
# ============================================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# Words to explore
words = [
    # Royalty
    "king",
    "queen",
    "prince",
    "princess",
    # Gender
    "man",
    "woman",
    "boy",
    "girl",
    # Animals
    "dog",
    "puppy",
    "cat",
    "kitten",
    # Geography
    "Paris",
    "France",
    "London",
    "England",
    # Technology
    "Python",
    "programming",
    "JavaScript",
    "coding",
]

print(f"\nEmbedding {len(words)} words...")
word_embeddings: ndarray = model.encode(words)
print(f"✓ Shape: {word_embeddings.shape}")

# Create a lookup dictionary
word_to_idx = {word: i for i, word in enumerate(words)}
idx_to_word = {i: word for i, word in enumerate(words)}


def get_embedding(word: str):
    """Get embedding for a word"""
    return word_embeddings[word_to_idx[word]]


# ============================================================================
# PART 2: Basic Similarity
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Word Similarity")
print("=" * 70)


def similarity(word1: str, word2: str):
    """Calculate cosine similarity between two words"""
    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)
    return cosine_similarity([emb1], [emb2])[0][0]


# Test pairs
word_pairs = [
    ("king", "queen"),
    ("king", "man"),
    ("Paris", "France"),
    ("Paris", "London"),
    ("dog", "cat"),
    ("dog", "puppy"),
    ("Python", "programming"),
    ("Python", "cat"),
]

print("\nSimilarity between word pairs:")
print(f"{'Word 1':<15} {'Word 2':<15} Similarity")
print("-" * 50)

for w1, w2 in word_pairs:
    sim = similarity(w1, w2)
    print(f"{w1:<15} {w2:<15} {sim:.4f}")

print("\n Higher score = more similar meaning!")

# ============================================================================
# PART 3: Find Most Similar Words
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Finding Similar Words")
print("=" * 70)


def most_similar(word: str, top_k: int=5, exclude_self=True):
    """
    Find most similar words to a given word.

    Args:
        word: The query word
        top_k: Number of results to return
        exclude_self: Whether to exclude the word itself

    Returns:
        List of (word, similarity_score) tuples
    """
    query_emb = get_embedding(word)
    similarities = cosine_similarity([query_emb], word_embeddings)[0]

    # Get sorted indices
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        w = idx_to_word[idx]
        if exclude_self and w == word:
            continue
        results.append((w, similarities[idx]))
        if len(results) >= top_k:
            break

    return results


# Test on several words
test_words = ["king", "Paris", "dog", "Python"]

for word in test_words:
    print(f"\nMost similar to '{word}':")
    similar = most_similar(word, top_k=5)
    for i, (w, score) in enumerate(similar, 1):
        print(f"  {i}. {w:<15} (similarity: {score:.4f})")

# ============================================================================
# PART 4: Vector Arithmetic - The Magic!
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Vector Arithmetic (Analogies)")
print("=" * 70)


def analogy(word1, word2, word3, top_k=5):
    """
    Solve analogy: word1 is to word2 as word3 is to ?

    Example: king is to man as queen is to ? → woman

    This works by: embedding(word1) - embedding(word2) + embedding(word3)
    """
    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)
    emb3 = get_embedding(word3)

    # Vector arithmetic
    result_emb = emb1 - emb2 + emb3

    # Find most similar word to result
    similarities = cosine_similarity([result_emb], word_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    # Exclude the input words from results
    exclude = {word1, word2, word3}
    results = []

    for idx in sorted_indices:
        w = idx_to_word[idx]
        if w not in exclude:
            results.append((w, similarities[idx]))
        if len(results) >= top_k:
            break

    return results


# Famous examples
print("\nFamous analogy: king - man + woman = ?")
print("(Should be close to 'queen')")
results = analogy("king", "man", "woman", top_k=5)
for i, (word, score) in enumerate(results, 1):
    marker = "✓" if word == "queen" else " "
    print(f"  {i}. {word:<15} (similarity: {score:.4f}) {marker}")

print("\nAnalogy: Paris - France + England = ?")
print("(Should be close to 'London')")
results = analogy("Paris", "France", "England", top_k=5)
for i, (word, score) in enumerate(results, 1):
    marker = "✓" if word == "London" else " "
    print(f"  {i}. {word:<15} (similarity: {score:.4f}) {marker}")

print("\nAnalogy: dog - puppy + kitten = ?")
print("(Should be close to 'cat')")
results = analogy("dog", "puppy", "kitten", top_k=5)
for i, (word, score) in enumerate(results, 1):
    marker = "✓" if word == "cat" else " "
    print(f"  {i}. {word:<15} (similarity: {score:.4f}) {marker}")

print("\n Vector arithmetic captures semantic relationships!")
print("   Relationships are directions in embedding space!")

# ============================================================================
# PART 5: Visualizing the Arithmetic
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Understanding the Math")
print("=" * 70)

# Let's break down the king - man + woman analogy
king_emb = get_embedding("king")
man_emb = get_embedding("man")
woman_emb = get_embedding("woman")
queen_emb = get_embedding("queen")

# Step by step
step1 = king_emb - man_emb  # Remove "maleness" from king
step2 = step1 + woman_emb  # Add "femaleness"

print("\nBreaking down: king - man + woman")
print("\n1. king embedding:")
print(f"   First 5 dimensions: {king_emb[:5]}")

print("\n2. Subtract 'man' (remove maleness):")
print(f"   First 5 dimensions: {step1[:5]}")

print("\n3. Add 'woman' (add femaleness):")
print(f"   First 5 dimensions: {step2[:5]}")

print("\n4. Compare to 'queen':")
print(f"   First 5 dimensions: {queen_emb[:5]}")

# Check similarity
sim_to_queen = cosine_similarity([step2], [queen_emb])[0][0]
print(f"\n5. Similarity to 'queen': {sim_to_queen:.4f}")

# ============================================================================
# PART 6: Distance Metrics Comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Cosine vs Euclidean Distance")
print("=" * 70)


def compare_metrics(word1, word2):
    """Compare different distance metrics"""
    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)

    # Cosine similarity (higher = more similar)
    cos_sim = cosine_similarity([emb1], [emb2])[0][0]

    # Euclidean distance (lower = more similar)
    euc_dist = euclidean_distances([emb1], [emb2])[0][0]

    # Dot product
    dot_prod = np.dot(emb1, emb2)

    return cos_sim, euc_dist, dot_prod


print("\nComparing metrics for word pairs:")
print(f"{'Word 1':<12} {'Word 2':<12} {'Cosine':<10} {'Euclidean':<12} {'Dot Product'}")
print("-" * 70)

for w1, w2 in [
    ("king", "queen"),
    ("king", "man"),
    ("Paris", "London"),
    ("dog", "cat"),
    ("Python", "cat"),
]:
    cos, euc, dot = compare_metrics(w1, w2)
    print(f"{w1:<12} {w2:<12} {cos:<10.4f} {euc:<12.4f} {dot:.4f}")

print("\n Cosine similarity is most common for embeddings")
print("   - Measures angle between vectors (direction)")
print("   - Range: -1 to 1 (usually 0 to 1 for text)")
print("   - Ignores magnitude, focuses on orientation")

# ============================================================================
# PART 7: Sentence-Level Analogies
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Sentence-Level Analogies")
print("=" * 70)

# Create sentence embeddings
sentences = [
    "I love programming in Python",
    "I enjoy coding in JavaScript",
    "I hate bugs in my code",
    "I dislike errors in my program",
    "The weather is sunny today",
    "It's raining outside now",
]

sentence_embeddings = model.encode(sentences)

print("\nTesting: Can we capture 'love → hate' relationship at sentence level?")

# love programming : hate programming :: enjoy coding : ?
love_prog = sentence_embeddings[0]  # I love programming
hate_bugs = sentence_embeddings[2]  # I hate bugs
enjoy_code = sentence_embeddings[1]  # I enjoy coding

# Calculate: (hate - love) + enjoy
result = hate_bugs - love_prog + enjoy_code

# Find closest
sims = cosine_similarity([result], sentence_embeddings)[0]
sorted_idx = np.argsort(sims)[::-1]

print("\nResult:")
for i, idx in enumerate(sorted_idx[:3], 1):
    print(f"  {i}. {sentences[idx][:50]}")
    print(f"     Similarity: {sims[idx]:.4f}")

# ============================================================================
# PART 8: Practical Applications
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: Practical Applications")
print("=" * 70)

print("""
Vector arithmetic enables many applications:

1. QUERY EXPANSION (Search)
   Original query: "Python"
   Expand to: ["Python", "programming", "coding"]
   → Better search results

2. RECOMMENDATION SYSTEMS
   User likes: ["Python", "programming"]
   Vector average → Find similar: "JavaScript", "coding"
   → Recommend related items

3. ANALOGY-BASED SEARCH
   "Show me to JavaScript what NumPy is to Python"
   → Find: JavaScript libraries similar to NumPy

4. RELATIONSHIP EXTRACTION
   king - man + woman = queen
   → Extract: gender relationship

5. SENTIMENT TRANSFER
   "I love this" - "love" + "hate" = "I hate this"
   → Generate contrasting examples

6. CONCEPT NAVIGATION
   Start: "Python"
   Add: +programming +advanced
   → Navigate to: "metaprogramming", "decorators"
""")

# ============================================================================
# PART 9: Build Your Own Analogies
# ============================================================================

print("\n" + "=" * 70)
print("PART 9: Try Your Own Analogies")
print("=" * 70)

print("""
Modify the code above to try your own analogies!

Examples to try:

1. Geography:
   - Tokyo - Japan + USA = ? (should be a US city)
   - Berlin - Germany + Italy = ?

2. Relationships:
   - brother - man + woman = sister
   - uncle - man + woman = aunt

3. Comparatives:
   - big - bigger + small = smaller
   - good - better + bad = worse

4. Professional:
   - doctor - hospital + school = teacher
   - chef - restaurant + courtroom = judge

5. Your own domain:
   - Add your own words
   - Find interesting relationships
   - Build domain-specific analogies

Tips:
- Not all analogies work perfectly
- Sentence embeddings can be more flexible than single words
- Try different embedding models for different results
""")

# ============================================================================
# PART 10: Advanced - Relationship Extraction
# ============================================================================

print("\n" + "=" * 70)
print("PART 10: Extracting Relationships")
print("=" * 70)


def extract_relationship(word1, word2):
    """Extract the 'relationship vector' between two words"""
    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)
    return emb2 - emb1


def apply_relationship(word, relationship_vector, top_k=5):
    """Apply a relationship to a word"""
    word_emb = get_embedding(word)
    result_emb = word_emb + relationship_vector

    similarities = cosine_similarity([result_emb], word_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices[:top_k]:
        w = idx_to_word[idx]
        if w != word:  # Exclude the input word
            results.append((w, similarities[idx]))

    return results


# Extract relationship: male → female
male_female = extract_relationship("man", "woman")

print("\nExtracted relationship: man → woman")
print("\nApplying to 'king':")
results = apply_relationship("king", male_female, top_k=5)
for word, score in results:
    print(f"  {word:<15} {score:.4f}")

print("\nApplying to 'boy':")
results = apply_relationship("boy", male_female, top_k=5)
for word, score in results:
    print(f"  {word:<15} {score:.4f}")

# Extract relationship: country → capital
country_capital = extract_relationship("France", "Paris")

print("\n\nExtracted relationship: France → Paris (country to capital)")
print("\nApplying to 'England':")
results = apply_relationship("England", country_capital, top_k=5)
for word, score in results:
    print(f"  {word:<15} {score:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
 Embeddings capture semantic relationships geometrically

 Vector arithmetic reveals relationships:
  - king - man + woman ≈ queen
  - Paris - France + England ≈ London

 Relationships are directions in embedding space
  - You can extract and reuse relationships
  - Apply them to new words

 Similarity metrics:
  - Cosine similarity: Most common (angle between vectors)
  - Euclidean distance: Absolute distance
  - Dot product: Raw similarity

 Applications:
  - Word analogies
  - Query expansion
  - Recommendation systems
  - Relationship extraction
  - Semantic navigation

 Limitations:
  - Analogies don't always work perfectly
  - Quality depends on training data
  - Some relationships are more geometric than others

 Next steps:
  - Run 05-rag-system.py to build a Q&A system
  - Run 06-visualization.py to SEE relationships in 2D
  - Experiment with your own words and relationships!
""")

print("\n" + "=" * 70)
print("✓ Similarity complete! Try 05-rag-system.py or 06-visualization.py")
print("=" * 70)
