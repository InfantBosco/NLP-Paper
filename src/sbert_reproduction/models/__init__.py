"""Models module for SBERT components.

Public API
----------
Pooling layers:
    MeanPooling, MaxPooling, CLSPooling, WeightedMeanPooling

Pooling utilities:
    normalize_embeddings, cosine_similarity

Encoder & sentence encoder:
    TransformerEncoderWrapper, TokenizerWrapper, SentenceEncoder

Similarity helpers:
    cosine_similarity_matrix, pairwise_encode, batch_encode

SBERT model + heads:
    SBERTModel, ClassificationHead, RegressionHead
"""

from .pooling import (
    MeanPooling,
    MaxPooling,
    CLSPooling,
    WeightedMeanPooling,
    normalize_embeddings,
    cosine_similarity,
)
from .encoder import TransformerEncoderWrapper, TokenizerWrapper
from .sentence_encoder import SentenceEncoder
from .similarity import cosine_similarity_matrix, pairwise_encode, batch_encode
from .sbert_model import SBERTModel, ClassificationHead, RegressionHead

__all__ = [
    # Pooling
    "MeanPooling",
    "MaxPooling",
    "CLSPooling",
    "WeightedMeanPooling",
    "normalize_embeddings",
    "cosine_similarity",
    # Encoder
    "TransformerEncoderWrapper",
    "TokenizerWrapper",
    "SentenceEncoder",
    # Similarity
    "cosine_similarity_matrix",
    "pairwise_encode",
    "batch_encode",
    # Model + heads
    "SBERTModel",
    "ClassificationHead",
    "RegressionHead",
]
