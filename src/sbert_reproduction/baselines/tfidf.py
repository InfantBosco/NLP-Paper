import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFBaseline:
    """TF-IDF vectorizer baseline with cosine similarity."""
    def __init__(self, max_features: int = 10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def predict_similarity(self, texts_a, texts_b):
        vecs_a = self.vectorizer.transform(texts_a)
        vecs_b = self.vectorizer.transform(texts_b)
        similarities = [cosine_similarity(vecs_a[i], vecs_b[i])[0][0] for i in range(len(texts_a))]
        return np.array(similarities)
