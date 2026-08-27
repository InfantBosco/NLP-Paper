"""
Pooling layers for Sentence-BERT reproduction.

Implements:
- MeanPooling   — attention-mask-aware mean over token sequence
- MaxPooling    — attention-mask-aware max over token sequence
- CLSPooling    — extract the [CLS] token representation
- WeightedMeanPooling — position-weighted mean pooling
- normalize_embeddings — L2-normalise embedding vectors
- cosine_similarity    — pairwise cosine similarity between two batches

Paper reference:
  Reimers & Gurevych (2019), Section 3 — Pooling Strategy.
  https://arxiv.org/abs/1908.10084

Official code reference:
  sentence-transformers/sentence_transformers/models/Pooling.py
  (independent re-implementation; official classes are NOT imported)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Mean Pooling
# ---------------------------------------------------------------------------

class MeanPooling(nn.Module):
    """
    Attention-mask-aware mean pooling over the token sequence.

    For each sentence the mean is taken only over non-padding tokens:

        u = (sum_t  mask_t * h_t) / max(sum_t mask_t, eps)

    where h_t is the token embedding and mask_t ∈ {0, 1}.

    Paper: Section 3 — "We use mean pooling … averaging all token embeddings
    while ignoring padding tokens."

    Args:
        token_embeddings: Tensor [batch, seq_len, hidden_dim]
        attention_mask:   Tensor [batch, seq_len]  — 1 for real, 0 for pad

    Returns:
        Tensor [batch, hidden_dim]
    """

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        # Expand mask to match embedding dimension
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        # Clamp denominator to avoid division by zero on all-padding rows
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


# ---------------------------------------------------------------------------
# Max Pooling
# ---------------------------------------------------------------------------

class MaxPooling(nn.Module):
    """
    Attention-mask-aware max pooling over the token sequence.

    Padding positions are replaced with −1e9 before taking the maximum
    so they never dominate the result.

    Paper: Section 3 — "We also experiment with max-over-time pooling."

    Args:
        token_embeddings: Tensor [batch, seq_len, hidden_dim]
        attention_mask:   Tensor [batch, seq_len]

    Returns:
        Tensor [batch, hidden_dim]
    """

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Use clone() so the original tensor is not mutated
        masked = token_embeddings.clone()
        masked[mask_expanded == 0] = -1e9
        return torch.max(masked, dim=1)[0]


# ---------------------------------------------------------------------------
# CLS Pooling
# ---------------------------------------------------------------------------

class CLSPooling(nn.Module):
    """
    CLS-token pooling: extracts the embedding at position 0.

    BERT encodes the [CLS] token at position 0; its representation
    is used as the aggregate sequence embedding for NLI fine-tuning
    in the baseline comparisons.

    Args:
        token_embeddings: Tensor [batch, seq_len, hidden_dim]
        attention_mask:   Tensor [batch, seq_len]  — not used, kept for API uniformity

    Returns:
        Tensor [batch, hidden_dim]
    """

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor = None) -> Tensor:
        return token_embeddings[:, 0, :]


# ---------------------------------------------------------------------------
# Weighted Mean Pooling
# ---------------------------------------------------------------------------

class WeightedMeanPooling(nn.Module):
    """
    Position-weighted mean pooling over the token sequence.

    Each token t receives a weight proportional to its position index + 1,
    then the weighted sum is divided by the total weight of non-padding tokens.

        u = (sum_t  (t+1) * mask_t * h_t) / max(sum_t (t+1) * mask_t, eps)

    This gives more weight to later tokens (can be useful for longer sentences
    where salient content appears near the end).

    Args:
        token_embeddings: Tensor [batch, seq_len, hidden_dim]
        attention_mask:   Tensor [batch, seq_len]

    Returns:
        Tensor [batch, hidden_dim]
    """

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        # Build position weights: [1, 2, ..., seq_len]
        device = token_embeddings.device
        position_weights = (
            torch.arange(1, seq_len + 1, dtype=torch.float, device=device)
            .unsqueeze(0)          # [1, seq_len]
            .expand(batch_size, -1)  # [batch, seq_len]
        )
        # Zero out padding positions
        masked_weights = position_weights * attention_mask.float()  # [batch, seq_len]
        # Expand weights to match hidden dim
        weights_expanded = masked_weights.unsqueeze(-1).expand_as(token_embeddings)  # [batch, seq_len, hidden]
        weighted_sum = torch.sum(token_embeddings * weights_expanded, dim=1)
        # Denominator: total weight of real tokens
        denom = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return weighted_sum / denom


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: Tensor, eps: float = 1e-12) -> Tensor:
    """
    L2-normalise each row of an embedding matrix so that every vector
    lies on the unit hypersphere.

    After normalisation, cosine similarity equals the dot product.

    Args:
        embeddings: Tensor [batch, dim]
        eps:        Small value to avoid division by zero

    Returns:
        Tensor [batch, dim]  — each row has unit L2-norm
    """
    return F.normalize(embeddings, p=2, dim=-1, eps=eps)


def cosine_similarity(u: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute element-wise cosine similarity between two batches of vectors.

        cos(u, v) = dot(u, v) / (||u|| * ||v||)

    Args:
        u:   Tensor [batch, dim]
        v:   Tensor [batch, dim]
        eps: Small constant for numerical stability

    Returns:
        Tensor [batch] — cosine similarity per pair, in [-1, 1]
    """
    return F.cosine_similarity(u, v, dim=-1, eps=eps)
