# embeddings_store.py
# In this module we manage the vector store that allows us to retrieve semantically similar reviews from our session history.
# We use sentence embeddings to represent each review as a numeric vectorand cosine similarity to find the closest matches.

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# We use a small but powerful multilingual embedding model
# that works well for both English and Spanish without extra configuration
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# We load the embedding model once at module level to avoid reloading
print('[INFO] We are loading the embedding model...')
embedder = SentenceTransformer(EMBEDDING_MODEL)
print('[INFO] Embedding model loaded.')

# We store all reviews and their embeddings in memory
# Each entry contains the text, its embedding vector and its sentiment label
_store = []


# We add a new review to our vector store after each analysis
def add_to_store(text, sentiment_label, confidence):
    # We compute the embedding vector for the new review
    embedding = embedder.encode(text, convert_to_numpy=True)

    _store.append({
        'text': text,
        'embedding': embedding,
        'sentiment': sentiment_label,
        'confidence': confidence
    })


# We retrieve the most semantically similar reviews from the store compared to the current input review
def retrieve_similar(text, top_k=2):
    # We need at least one stored review to do retrieval
    if not _store:
        return []

    # We compute the embedding of the current input
    query_embedding = embedder.encode(text, convert_to_numpy=True).reshape(1, -1)

    # We stack all stored embeddings into a matrix for batch comparison
    stored_embeddings = np.vstack([entry['embedding'] for entry in _store])

    # We compute cosine similarity between the query and all stored reviews
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    # We get the indices of the top_k most similar reviews
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # We return the most similar entries along with their similarity scores
    results = []
    for idx in top_indices:
        results.append({
            'text': _store[idx]['text'],
            'sentiment': _store[idx]['sentiment'],
            'confidence': _store[idx]['confidence'],
            'similarity': round(float(similarities[idx]) * 100, 2)
        })

    return results


# We return the number of reviews currently stored
def store_size():
    return len(_store)