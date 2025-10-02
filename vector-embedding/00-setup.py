"""
00 - Setup & Model Download
============================

Run this FIRST to download the embedding model and verify everything works.

This will take a few minutes on first run as it downloads the model (~80MB).
"""

import importlib.util

print("=" * 70)
print("SETUP - Downloading Embedding Model")
print("=" * 70)

print("\n1. Testing imports...")
try:
    assert importlib.util.find_spec("numpy") is not None

    print("   ✓ numpy")
except ImportError as e:
    print(f"   ✗ numpy - {e}")
    print("   Install: pip install numpy")
    exit(1)

try:
    assert importlib.util.find_spec("sklearn") is not None

    print("   ✓ scikit-learn")
except ImportError as e:
    print(f"   ✗ scikit-learn - {e}")
    print("   Install: pip install scikit-learn")
    exit(1)

try:
    assert importlib.util.find_spec("torch") is not None

    print("   ✓ torch")
except ImportError as e:
    print(f"   ✗ torch - {e}")
    print("   Install: pip install torch")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer

    print("   ✓ sentence-transformers")
except ImportError as e:
    print(f"   ✗ sentence-transformers - {e}")
    print("   Install: pip install sentence-transformers")
    exit(1)

print("\n2. Downloading embedding model (this may take a few minutes)...")
print("   Model: all-MiniLM-L6-v2 (384 dimensions, ~80MB)")
print("   This only happens once - it will be cached locally.\n")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("   ✓ Model loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

print("\n3. Testing embedding generation...")
test_text = "Hello, this is a test sentence."
try:
    embedding = model.encode(test_text)
    print(f"   ✓ Generated embedding: shape {embedding.shape}")
    print(f"   ✓ First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"   ✗ Error generating embedding: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✓ SETUP COMPLETE!")
print("=" * 70)
print("\nYou're ready to run the examples!")
print("\nNext steps:")
print("  python vector-embedding/01-basics.py")
print("  python vector-embedding/02-semantic-search.py")
print("  python vector-embedding/03-clustering.py")
print("\nHave fun!")
