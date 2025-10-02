"""
02 - Semantic Search Engine
============================

Build a search engine that understands MEANING, not just keywords.

Real-world use cases:
- Search documentation intelligently
- Find relevant emails/messages
- Product search that understands intent
- Code search by functionality
- Research paper discovery

This is the foundation for RAG systems!
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

print("=" * 70)
print("SEMANTIC SEARCH ENGINE")
print("=" * 70)

# ============================================================================
# PART 1: The Document Collection
# ============================================================================

# Let's create a knowledge base of Python documentation snippets
documents = [
    "Lists are mutable sequences, typically used to store collections of homogeneous items",
    "Dictionaries are mutable mappings from keys to values, implemented as hash tables",
    "Sets are unordered collections of unique elements, useful for membership testing",
    "Tuples are immutable sequences, often used to store heterogeneous data",
    "Strings are immutable sequences of Unicode characters, used for text data",
    "NumPy arrays provide efficient storage and operations on multi-dimensional numeric data",
    "Pandas DataFrames are 2D labeled data structures with columns of potentially different types",
    "Functions are defined using the def keyword and can accept parameters",
    "Classes are templates for creating objects, defined with the class keyword",
    "Decorators are a way to modify or enhance functions or classes without changing their code",
    "List comprehensions provide a concise way to create lists based on existing sequences",
    "Lambda functions are small anonymous functions defined with the lambda keyword",
    "Generators yield values one at a time and are memory efficient for large sequences",
    "Context managers handle resource setup and cleanup automatically using with statements",
    "Exception handling uses try-except blocks to gracefully handle runtime errors",
]

print(f"\nDocument collection: {len(documents)} Python docs snippets")

# ============================================================================
# PART 2: Build the Search Index
# ============================================================================

print("\n" + "=" * 70)
print("INDEXING DOCUMENTS")
print("=" * 70)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all documents (this is the "indexing" step)
print(f"\nEmbedding {len(documents)} documents...")
doc_embeddings = model.encode(documents, show_progress_bar=True)
print(f"✓ Index built! Shape: {doc_embeddings.shape}")
print(f"  Each document → {doc_embeddings.shape[1]}-dimensional vector")

# ============================================================================
# PART 3: Semantic Search
# ============================================================================


def semantic_search(query: str, top_k: int=3):
    """
    Search for documents similar to the query.

    Args:
        query: Search query (natural language)
        top_k: Number of results to return

    Returns:
        List of (doc_index, similarity_score) tuples
    """
    # Embed the query
    query_embedding = model.encode(query)

    # Calculate similarities to all documents
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Get top-k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(idx, similarities[idx]) for idx in top_indices]
    return results


# ============================================================================
# PART 4: Test Different Queries
# ============================================================================

print("\n" + "=" * 70)
print("SEMANTIC SEARCH IN ACTION")
print("=" * 70)

test_queries = [
    "How do I store key-value pairs?",
    "What's good for unique items without duplicates?",
    "I need to process large amounts of numbers efficiently",
    "How can I create a list from another list quickly?",
    "What should I use for read-only data that won't change?",
]

for query in test_queries:
    print(f"\n{'─' * 70}")
    print(f'Query: "{query}"')
    print(f"{'─' * 70}")

    results = semantic_search(query, top_k=3)

    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n{rank}. [Score: {score:.4f}]")
        print(f"   {documents[idx]}")

# ============================================================================
# PART 5: Keyword Search vs Semantic Search
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: Keyword vs Semantic Search")
print("=" * 70)


def keyword_search(query: str, docs: List[str], top_k: int=3):
    """
    Simple keyword search (for comparison).
    Returns documents that contain query words.
    """
    query_words = set(query.lower().split())
    scores = []

    for doc in docs:
        doc_words = set(doc.lower().split())
        # Score = number of matching words
        score = len(query_words & doc_words)
        scores.append(score)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
    return results


# Test query where semantic search shines
query = "How do I store pairs of information together?"

print(f'\nQuery: "{query}"')
print(f"\n{'Keyword Search Results:':<40}")
print("─" * 70)
keyword_results = keyword_search(query, documents, top_k=3)
if keyword_results:
    for rank, (idx, score) in enumerate(keyword_results, 1):
        print(f"{rank}. [Matches: {score} words] {documents[idx][:60]}...")
else:
    print("No results found!")

print(f"\n{'Semantic Search Results:':<40}")
print("─" * 70)
semantic_results = semantic_search(query, top_k=3)
for rank, (idx, score) in enumerate(semantic_results, 1):
    print(f"{rank}. [Score: {score:.4f}] {documents[idx][:60]}...")

print("\ Notice: Semantic search found 'Dictionaries' and 'Tuples'")
print("   even though the query didn't contain those exact words!")

# ============================================================================
# PART 6: Interactive Search
# ============================================================================

print("\n" + "=" * 70)
print("INTERACTIVE SEARCH")
print("=" * 70)


def show_search_results(query: str, top_k: int=5):
    """Pretty print search results"""
    print(f'\nSearching for: "{query}"')
    print("─" * 70)

    results = semantic_search(query, top_k=top_k)

    if not results:
        print("No results found.")
        return

    for rank, (idx, score) in enumerate(results, 1):
        # Color code by relevance
        if score > 0.5:
            emoji = "[High]"  # Highly relevant
        elif score > 0.3:
            emoji = "[Moderate]"  # Moderately relevant
        else:
            emoji = "[Less]"  # Less relevant

        print(f"\n{rank}. {emoji} Relevance: {score:.4f}")
        print(f"   {documents[idx]}")


# Try some searches
example_searches = [
    "working with tabular data",
    "immutable data structures",
    "handling errors in code",
    "memory efficient data processing",
]

for search_query in example_searches:
    show_search_results(search_query, top_k=3)

# ============================================================================
# PART 7: Build Your Own Search Engine
# ============================================================================

print("\n" + "=" * 70)
print("BUILD YOUR OWN")
print("=" * 70)

print("""
Now you can build a search engine for any domain!

Example applications:

1. DOCUMENTATION SEARCH
   documents = [... your API docs ...]
   query = "how to authenticate users?"

2. EMAIL SEARCH
   documents = [... email bodies ...]
   query = "find emails about the product launch"

3. CODE SEARCH
   documents = [... function docstrings ...]
   query = "function to parse JSON"

4. PRODUCT SEARCH
   documents = [... product descriptions ...]
   query = "comfortable running shoes for beginners"

5. RESEARCH PAPERS
   documents = [... paper abstracts ...]
   query = "papers about attention mechanisms in transformers"

Template code:
""")

print("""
# Your custom search engine
my_documents = [
    # Add your documents here
]

# Index them
doc_embeddings = model.encode(my_documents)

# Search!
def search(query):
    query_emb = model.encode(query)
    sims = cosine_similarity([query_emb], doc_embeddings)[0]
    top_idx = np.argmax(sims)
    return my_documents[top_idx]

result = search("your query here")
print(result)
""")

# ============================================================================
# PART 8: Performance Considerations
# ============================================================================

print("\n" + "=" * 70)
print("PERFORMANCE TIPS")
print("=" * 70)

print("""
For larger collections:

1. BATCH INDEXING
   # Embed in batches for speed
   embeddings = model.encode(documents,
                             batch_size=32,
                             show_progress_bar=True)

2. SAVE THE INDEX
   # Don't re-embed every time!
   import pickle
   with open('embeddings.pkl', 'wb') as f:
       pickle.dump(doc_embeddings, f)

   # Load later
   with open('embeddings.pkl', 'rb') as f:
       doc_embeddings = pickle.load(f)

3. USE VECTOR DATABASES (for 1000s+ documents)
   - FAISS (Facebook)
   - Pinecone
   - Weaviate
   - Qdrant
   See 10-production-rag.py for examples

4. APPROXIMATE NEAREST NEIGHBORS
   # For millions of documents, use ANN algorithms
   # See example in 10-production-rag.py
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
Semantic search finds documents by MEANING, not keywords

  Process:
   1. Embed all documents once (indexing)
   2. Embed the query
   3. Calculate cosine similarity
   4. Return top results

  Better than keyword search because:
   - Understands synonyms ("buy" = "purchase")
   - Understands paraphrasing
   - Handles conceptual queries
   - Works across languages (with multilingual models)

  This is the foundation for:
   - RAG systems (next: 05-rag-system.py)
   - Recommendation engines
   - Question answering
   - Document retrieval

  Next steps:
   - Run 03-clustering.py to auto-organize documents
   - Run 05-rag-system.py to add LLM-generated answers
   - Run 10-production-rag.py for production-scale systems
""")

print("\n" + "=" * 70)
print("✓ Semantic search complete! Ready for 03-clustering.py")
print("=" * 70)
