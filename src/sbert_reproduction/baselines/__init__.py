"""Baseline models module.

Baselines (independent reproduction):
    TFIDFBaseline              — TF-IDF + cosine similarity
    AveragedWordEmbeddingsBaseline — averaged GloVe / random vectors
    VanillaBERTCLSBaseline     — un-finetuned BERT, CLS pooling
    VanillaBERTMeanBaseline    — un-finetuned BERT, mean pooling

External comparison only (uses official sentence-transformers library):
    SBERTReferenceBaseline     — official SBERT model, NOT part of reproduction
"""

from .tfidf import TFIDFBaseline
from .averaged_embeddings import AveragedWordEmbeddingsBaseline
from .bert_cls import VanillaBERTCLSBaseline
from .bert_mean import VanillaBERTMeanBaseline
from .sbert_reference import SBERTReferenceBaseline

__all__ = [
    "TFIDFBaseline",
    "AveragedWordEmbeddingsBaseline",
    "VanillaBERTCLSBaseline",
    "VanillaBERTMeanBaseline",
    "SBERTReferenceBaseline",
]
