"""
Cosine similarity utilities and sentence-pair encoding helpers.

Provides:
- cosine_similarity_matrix  — all-pairs cosine similarity
- pairwise_encode           — encode sentence pairs with shared encoder
- batch_encode              — encode a list of sentences in batches

Paper reference:
  Reimers & Gurevych (2019), Section 3 — "cosine-similarity can then be
  used to find semantically similar sentences."

Official code reference:
  sentence-transformers/sentence_transformers/util.py
  (re-implemented independently; no official classes are imported)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple


def cosine_similarity_matrix(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute the full cosine similarity matrix between two sets of vectors.

        S[i, j] = cos(a_i, b_j)

    Args:
        a:   Tensor [m, dim]
        b:   Tensor [n, dim]
        eps: Small constant added to norms for numerical stability.

    Returns:
        Tensor [m, n]  — values in [-1, 1]
    """
    a_norm = F.normalize(a, p=2, dim=-1, eps=eps)   # [m, dim]
    b_norm = F.normalize(b, p=2, dim=-1, eps=eps)   # [n, dim]
    return torch.mm(a_norm, b_norm.t())             # [m, n]


def pairwise_encode(
    encoder,
    features_a: dict,
    features_b: dict,
) -> Tuple[Tensor, Tensor]:
    """
    Encode a pair of sentence batches with a shared :class:`SentenceEncoder`.

    Both sentence batches are forwarded through the *same* encoder (shared
    weights — the Siamese architecture from the paper).

    Args:
        encoder:    :class:`SentenceEncoder` instance.
        features_a: Dict with ``input_ids``, ``attention_mask``
                    (and optionally ``token_type_ids``) for sentence A.
        features_b: Same structure for sentence B.

    Returns:
        Tuple ``(u, v)`` where u = enc(A) and v = enc(B),
        both Tensor [batch, hidden_dim].
    """
    u = encoder(
        input_ids=features_a["input_ids"],
        attention_mask=features_a["attention_mask"],
        token_type_ids=features_a.get("token_type_ids"),
    )
    v = encoder(
        input_ids=features_b["input_ids"],
        attention_mask=features_b["attention_mask"],
        token_type_ids=features_b.get("token_type_ids"),
    )
    return u, v


def batch_encode(
    encoder,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
) -> Tensor:
    """
    Encode a list of raw sentences into embeddings using the encoder.

    Each sentence is tokenised via *tokenizer*, then forwarded through
    *encoder* in eval + no_grad mode.

    Args:
        encoder:    :class:`SentenceEncoder` instance.
        tokenizer:  :class:`TokenizerWrapper` instance.
        texts:      List of raw strings.
        batch_size: Mini-batch size for encoding.

    Returns:
        Tensor [len(texts), hidden_dim]  on CPU.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list of strings.")

    return encoder.encode_text(tokenizer, texts, batch_size=batch_size)
