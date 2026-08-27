"""
SentenceEncoder — combines TransformerEncoderWrapper + a Pooling layer
to output a fixed-size sentence embedding vector.

This is the core building block of the Siamese / triplet network
described in Reimers & Gurevych (2019).

Paper reference:
  Section 3 — "We use mean pooling of all output vectors."

Official code reference:
  sentence-transformers/sentence_transformers/SentenceTransformer.py
  (re-implemented independently; no official classes are imported)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List, Optional
from torch import Tensor

from .pooling import (
    MeanPooling,
    MaxPooling,
    CLSPooling,
    WeightedMeanPooling,
    normalize_embeddings,
)


POOLING_MODES = ("mean", "max", "cls", "weightedmean")


class SentenceEncoder(nn.Module):
    """
    Combines a transformer encoder backbone with a pooling strategy to
    produce a fixed-size sentence embedding.

    Args:
        encoder:        A :class:`TransformerEncoderWrapper` (or any nn.Module
                        whose forward returns [batch, seq_len, hidden_dim]).
        pooling_mode:   One of ``"mean"``, ``"max"``, ``"cls"``,
                        ``"weightedmean"``.
        normalize:      If True, L2-normalise output embeddings.

    Example::

        encoder = TransformerEncoderWrapper("bert-base-uncased")
        model   = SentenceEncoder(encoder, pooling_mode="mean", normalize=True)
        emb     = model(input_ids, attention_mask)   # [B, 768]
    """

    def __init__(
        self,
        encoder: nn.Module,
        pooling_mode: str = "mean",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.pooling_mode = pooling_mode.lower()

        if self.pooling_mode == "mean":
            self.pooling: nn.Module = MeanPooling()
        elif self.pooling_mode == "max":
            self.pooling = MaxPooling()
        elif self.pooling_mode == "cls":
            self.pooling = CLSPooling()
        elif self.pooling_mode == "weightedmean":
            self.pooling = WeightedMeanPooling()
        else:
            raise ValueError(
                f"Unknown pooling_mode '{pooling_mode}'. "
                f"Choose one of {POOLING_MODES}."
            )

    # ------------------------------------------------------------------
    @contextmanager
    def inference_mode(self):
        """
        Context manager: switches to eval mode with torch.no_grad().

        Usage::

            with sentence_encoder.inference_mode():
                embeddings = sentence_encoder(input_ids, attention_mask)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            yield
        self.train(was_training)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode a batch of tokenised sentences into fixed-size vectors.

        Args:
            input_ids:      LongTensor [batch, seq_len]
            attention_mask: LongTensor [batch, seq_len]
            token_type_ids: LongTensor [batch, seq_len]  (optional, BERT)

        Returns:
            Tensor [batch, hidden_dim]
        """
        token_embeddings = self.encoder(input_ids, attention_mask, token_type_ids)
        sentence_embedding = self.pooling(token_embeddings, attention_mask)
        if self.normalize:
            sentence_embedding = normalize_embeddings(sentence_embedding)
        return sentence_embedding

    # ------------------------------------------------------------------
    def encode_text(
        self,
        tokenizer,
        texts: List[str],
        batch_size: int = 32,
    ) -> Tensor:
        """
        High-level helper: tokenise *texts* with *tokenizer* and return
        sentence embeddings.

        Operates in inference mode (no gradients, eval).

        Args:
            tokenizer:  A :class:`TokenizerWrapper` instance.
            texts:      List of raw strings.
            batch_size: Number of sentences per forward pass.

        Returns:
            Tensor [len(texts), hidden_dim]
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings.")

        all_embeddings: List[Tensor] = []
        with self.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                features = tokenizer.tokenize(batch_texts)
                emb = self.forward(
                    input_ids=features["input_ids"],
                    attention_mask=features["attention_mask"],
                    token_type_ids=features.get("token_type_ids"),
                )
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)
