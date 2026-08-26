import numpy as np

class AveragedWordEmbeddingsBaseline:
    """Average word embeddings baseline (e.g. GloVe 300d)."""
    def __init__(self, embedding_dict=None, dim: int = 300):
        self.embedding_dict = embedding_dict or {}
        self.dim = dim

    def encode_sentence(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vectors = [self.embedding_dict[tok] for tok in tokens if tok in self.embedding_dict]
        if not vectors:
            return np.zeros(self.dim)
        return np.mean(vectors, axis=0)

    def predict_similarity(self, texts_a, texts_b):
        sims = []
        for ta, tb in zip(texts_a, texts_b):
            va = self.encode_sentence(ta)
            vb = self.encode_sentence(tb)
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a == 0 or norm_b == 0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(va, vb) / (norm_a * norm_b)))
        return np.array(sims)
