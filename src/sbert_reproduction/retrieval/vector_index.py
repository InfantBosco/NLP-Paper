import numpy as np
from .brute_force import BruteForceSearch

class FlatVectorIndex:
    """Simple in-memory vector index wrapper."""
    def __init__(self, embeddings: np.ndarray):
        self.search_engine = BruteForceSearch(embeddings)

    def query(self, query_vec: np.ndarray, top_k: int = 5):
        return self.search_engine.search(query_vec, top_k=top_k)
