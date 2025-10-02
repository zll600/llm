"""
05 - RAG System (Retrieval-Augmented Generation)
=================================================

Build a system that can answer questions using your documents.

RAG = Retrieval + Generation:
1. Retrieve relevant documents using embeddings (what we learned in 02)
2. Generate answers using an LLM with that context

Real-world use cases:
- Chat with your PDF documents
- Q&A over company knowledge base
- Customer support chatbots with accurate info
- Research assistants
- Technical documentation chat

This example shows the core RAG pattern (without requiring API keys).
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("=" * 70)
print("RAG SYSTEM - Chat with Your Documents")
print("=" * 70)

# ============================================================================
# PART 1: Knowledge Base
# ============================================================================

# Our knowledge base about Python and Machine Learning
knowledge_base = [
    {
        "id": "doc_001",
        "title": "Python Lists",
        "content": "Lists in Python are mutable, ordered sequences. You can create a list using square brackets, like my_list = [1, 2, 3]. Lists support indexing, slicing, appending, and many other operations. They can contain mixed types and duplicate values.",
    },
    {
        "id": "doc_002",
        "title": "Python Dictionaries",
        "content": "Dictionaries are key-value pair data structures in Python. Create them with curly braces: my_dict = {'name': 'Alice', 'age': 30}. They provide O(1) average-case lookup time and are perfect for mapping relationships.",
    },
    {
        "id": "doc_003",
        "title": "Machine Learning Basics",
        "content": "Machine learning is a subset of AI where systems learn from data. There are three main types: supervised learning (with labeled data), unsupervised learning (finding patterns without labels), and reinforcement learning (learning through rewards).",
    },
    {
        "id": "doc_004",
        "title": "Neural Networks",
        "content": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons). Each connection has a weight that's adjusted during training. Deep learning refers to neural networks with many layers.",
    },
    {
        "id": "doc_005",
        "title": "Python Functions",
        "content": "Functions in Python are defined using the def keyword. They can accept parameters, return values, and have default arguments. Functions help organize code into reusable blocks. Lambda functions provide a way to create small anonymous functions.",
    },
    {
        "id": "doc_006",
        "title": "Supervised Learning",
        "content": "Supervised learning uses labeled training data to learn mappings from inputs to outputs. Common algorithms include linear regression (for continuous values), logistic regression (for classification), decision trees, and neural networks. The goal is to generalize to unseen data.",
    },
    {
        "id": "doc_007",
        "title": "Training Neural Networks",
        "content": "Training a neural network involves forward propagation (making predictions), calculating loss (error), and backpropagation (adjusting weights). This process uses optimization algorithms like gradient descent. Hyperparameters like learning rate affect training speed and quality.",
    },
    {
        "id": "doc_008",
        "title": "Python Decorators",
        "content": "Decorators are a powerful feature that allows you to modify or enhance functions without changing their code. They use the @decorator syntax. Common use cases include logging, timing, authentication, and caching. Decorators take a function and return a modified version.",
    },
]

print(f"\nKnowledge Base: {len(knowledge_base)} documents")
print("\nSample documents:")
for doc in knowledge_base[:2]:
    print(f"  [{doc['id']}] {doc['title']}")
    print(f"      {doc['content'][:60]}...")

# ============================================================================
# PART 2: Index the Knowledge Base
# ============================================================================

print("\n" + "=" * 70)
print("INDEXING KNOWLEDGE BASE")
print("=" * 70)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract and embed the content
documents_text = [doc["content"] for doc in knowledge_base]

print(f"\nEmbedding {len(documents_text)} documents...")
doc_embeddings = model.encode(documents_text, show_progress_bar=True)
print(f"✓ Index created! Shape: {doc_embeddings.shape}")

# ============================================================================
# PART 3: Retrieval Function
# ============================================================================


def retrieve_relevant_docs(query: str, top_k: int=3):
    """
    Retrieve the most relevant documents for a query.

    This is the 'R' in RAG - Retrieval.

    Args:
        query: User's question
        top_k: Number of documents to retrieve

    Returns:
        List of relevant documents with scores
    """
    # Embed the query
    query_embedding = model.encode(query)

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return documents with scores
    results = []
    for idx in top_indices:
        results.append({"document": knowledge_base[idx], "score": similarities[idx]})

    return results


# ============================================================================
# PART 4: Simple Generation (Template-Based)
# ============================================================================


def generate_answer_template(query: str, retrieved_docs: str):
    """
    Generate an answer using retrieved documents.

    This is a simplified 'G' in RAG - Generation.
    In production, you'd use an LLM (GPT, Claude, Llama, etc.)

    For now, we'll use a simple template.
    """
    # Build context from retrieved documents
    context_parts = []
    for i, item in enumerate(retrieved_docs, 1):
        doc = item["document"]
        score = item["score"]
        context_parts.append(f"[Source {i}] {doc['title']}: {doc['content']}")

    context = "\n\n".join(context_parts)

    # Simple template-based response
    answer = f"""
Based on the retrieved information, here's what I found:

{context}

To fully answer "{query}", you would want to focus on the most relevant source above (highest score).
"""

    return answer


# ============================================================================
# PART 5: Complete RAG Pipeline
# ============================================================================


def rag_query(question: str, top_k: int=3, show_sources: bool=True):
    """
    Complete RAG pipeline: Retrieve + Generate

    Args:
        question: User's question
        top_k: Number of docs to retrieve
        show_sources: Whether to show retrieved documents
    """
    print(f"\n{'=' * 70}")
    print(f"Question: {question}")
    print(f"{'=' * 70}")

    # Step 1: Retrieve
    print("\n[RETRIEVAL] Finding relevant documents...")
    retrieved = retrieve_relevant_docs(question, top_k=top_k)

    if show_sources:
        print(f"\nRetrieved {len(retrieved)} documents:\n")
        for i, item in enumerate(retrieved, 1):
            doc = item["document"]
            score = item["score"]
            relevance = (
                "High" if score > 0.5 else "Medium" if score > 0.3 else "Low"
            )
            print(f"{i}. [{doc['id']}] {doc['title']}")
            print(f"   Relevance: {relevance} ({score:.4f})")
            print(f"   {doc['content'][:100]}...\n")

    # Step 2: Generate
    print("[GENERATION] Creating answer...\n")
    answer = generate_answer_template(question, retrieved)

    print(f"{'─' * 70}")
    print("ANSWER:")
    print(f"{'─' * 70}")
    print(answer)

    return answer


# ============================================================================
# PART 6: Example Queries
# ============================================================================

print("\n" + "=" * 70)
print("RAG IN ACTION")
print("=" * 70)

example_questions = [
    "What are the different types of machine learning?",
    "How do I create a dictionary in Python?",
    "How does neural network training work?",
    "What are Python decorators used for?",
]

# Run first example in detail
rag_query(example_questions[0], top_k=2, show_sources=True)

# Show other examples more concisely
print("\n\n" + "=" * 70)
print("MORE EXAMPLES")
print("=" * 70)

for question in example_questions[1:]:
    retrieved = retrieve_relevant_docs(question, top_k=1)
    doc = retrieved[0]["document"]
    score = retrieved[0]["score"]

    print(f"\nQ: {question}")
    print(f"A: Retrieved '{doc['title']}' (score: {score:.4f})")
    print(f"   {doc['content'][:120]}...")

# ============================================================================
# PART 7: RAG with Actual LLM (Optional)
# ============================================================================

print("\n\n" + "=" * 70)
print("USING A REAL LLM (Optional)")
print("=" * 70)

print("""
To use a real LLM for generation, you have several options:

1. OPENAI API
   ```python
   import openai

   def generate_with_gpt(query, context):
       prompt = f'''
       Context: {context}

       Question: {query}

       Answer the question using only the context provided.
       '''

       response = openai.ChatCompletion.create(
           model="gpt-4",
           messages=[{"role": "user", "content": prompt}]
       )

       return response.choices[0].message.content
   ```

2. LOCAL LLM (Ollama, LLaMA, etc.)
   ```python
   import requests

   def generate_with_ollama(query, context):
       prompt = f"Context: {context}\\n\\nQuestion: {query}"

       response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama2',
                                   'prompt': prompt
                               })

       return response.json()['response']
   ```

3. HUGGING FACE TRANSFORMERS
   ```python
   from transformers import pipeline

   generator = pipeline('text-generation', model='gpt2')

   def generate_with_hf(query, context):
       prompt = f"Context: {context}\\nQuestion: {query}\\nAnswer:"
       response = generator(prompt, max_length=200)
       return response[0]['generated_text']
   ```

For production, see 10-production-rag.py!
""")

# ============================================================================
# PART 8: RAG Best Practices
# ============================================================================

print("\n" + "=" * 70)
print("RAG BEST PRACTICES")
print("=" * 70)

print("""
1. CHUNKING STRATEGY
   - Don't embed entire documents
   - Split into chunks (e.g., 500 tokens)
   - Overlap chunks for context (e.g., 50 tokens)

   Example:
   ```python
   def chunk_document(text, chunk_size=500, overlap=50):
       words = text.split()
       chunks = []
       for i in range(0, len(words), chunk_size - overlap):
           chunk = ' '.join(words[i:i + chunk_size])
           chunks.append(chunk)
       return chunks
   ```

2. METADATA FILTERING
   - Store metadata with embeddings (date, author, category)
   - Pre-filter before similarity search
   - E.g., "Only search documents from 2024"

3. HYBRID SEARCH
   - Combine semantic search (embeddings) with keyword search
   - Use both for better results

4. RE-RANKING
   - Retrieve more candidates (e.g., top 20)
   - Re-rank using a more sophisticated model
   - Return top 3 after re-ranking

5. CONTEXT WINDOW OPTIMIZATION
   - LLMs have token limits (e.g., 4k, 8k, 32k)
   - Don't waste tokens on irrelevant context
   - Summarize retrieved docs if needed

6. CITATION/SOURCE TRACKING
   - Always track which documents were used
   - Include citations in the answer
   - Allows users to verify information

7. EVALUATION
   - Test retrieval quality (are right docs retrieved?)
   - Test generation quality (are answers accurate?)
   - Use metrics like MRR, NDCG, F1
""")

# ============================================================================
# PART 9: Build Your Own RAG
# ============================================================================

print("\n" + "=" * 70)
print("BUILD YOUR OWN RAG SYSTEM")
print("=" * 70)

print("""
Step-by-step guide:

1. PREPARE YOUR DATA
   ```python
   # Load your documents
   docs = load_documents_from_folder("./my_docs/")

   # Chunk them
   chunks = []
   for doc in docs:
       chunks.extend(chunk_document(doc, chunk_size=500))
   ```

2. CREATE EMBEDDINGS INDEX
   ```python
   # Embed chunks
   embeddings = model.encode(chunks, show_progress_bar=True)

   # Save for later
   import pickle
   with open('my_index.pkl', 'wb') as f:
       pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
   ```

3. BUILD RETRIEVAL FUNCTION
   ```python
   def search(query, top_k=5):
       query_emb = model.encode(query)
       scores = cosine_similarity([query_emb], embeddings)[0]
       top_idx = np.argsort(scores)[::-1][:top_k]
       return [chunks[i] for i in top_idx]
   ```

4. INTEGRATE LLM
   ```python
   def answer_question(question):
       context = search(question, top_k=3)
       context_str = "\\n\\n".join(context)

       # Use your LLM of choice
       answer = call_llm(question, context_str)
       return answer
   ```

5. ADD UI (optional)
   - Streamlit for quick web UI
   - Gradio for ML demos
   - FastAPI for REST API
   - Custom web app

Examples in the wild:
- ChatPDF
- Notion AI
- GitHub Copilot Chat
- Perplexity AI
- And thousands more...
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
 RAG = Retrieval-Augmented Generation

 Pipeline:
  1. User asks a question
  2. Retrieve relevant documents (using embeddings)
  3. Pass question + context to LLM
  4. LLM generates answer based on retrieved context

 Advantages over pure LLM:
  - Access to private/recent data
  - Reduces hallucinations
  - Can cite sources
  - No need to fine-tune LLM
  - Easy to update knowledge base

 Key components:
  - Embedding model (for retrieval)
  - Vector search (cosine similarity or vector DB)
  - LLM (for generation)
  - Document chunking strategy

 Next steps:
  - Run 10-production-rag.py for production-scale RAG
  - Run 06-visualization.py to see embeddings
  - Integrate with actual LLM (OpenAI, Ollama, etc.)
  - Add your own documents!
""")

print("\n" + "=" * 70)
print("✓ RAG system complete! Try 06-visualization.py or 10-production-rag.py")
print("=" * 70)
