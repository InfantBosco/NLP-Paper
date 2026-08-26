import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BruteForceSearch:
    def __init__(self, corpus_embeddings: np.ndarray):
        self.embeddings = corpus_embeddings

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.reshape(1, -1)
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return top_indices, sims[top_indices]
