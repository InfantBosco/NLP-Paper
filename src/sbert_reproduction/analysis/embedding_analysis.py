import numpy as np

def compute_embedding_norms(embeddings: np.ndarray) -> dict:
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms))
    }
