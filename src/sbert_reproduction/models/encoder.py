"""
Transformer encoder wrapper for Sentence-BERT reproduction.

Wraps a HuggingFace AutoModel backbone without importing any official
sentence-transformers classes.  Also provides a TokenizerWrapper that
converts raw strings into model-ready tensors.

Paper reference:
  Reimers & Gurevych (2019), Section 3 — we use "BERT / RoBERTa as the
  fixed pre-trained network and fine-tune it."
  https://arxiv.org/abs/1908.10084

Official code reference:
  sentence-transformers/sentence_transformers/models/Transformer.py
  (re-implemented independently)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from contextlib import contextmanager
from typing import List, Optional


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """
    Thin wrapper around a HuggingFace tokenizer.

    Converts a list of strings into a dict suitable for passing to
    :class:`TransformerEncoderWrapper.forward`.

    Args:
        model_name:      HuggingFace model identifier (e.g. "bert-base-uncased")
        max_seq_length:  Maximum number of tokens.  Sequences are truncated
                         and padded to this length.
        device:          Target device ("cpu" or "cuda").
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_seq_length: int = 128,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device = device
        self._tokenizer = None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # ------------------------------------------------------------------
    def tokenize(self, texts: List[str]) -> dict:
        """
        Tokenise *texts* and return a dict of tensors ready for the encoder.

        Returns:
            {
              "input_ids":      LongTensor [batch, seq_len],
              "attention_mask": LongTensor [batch, seq_len],
              "token_type_ids": LongTensor [batch, seq_len]  (BERT only),
            }
        """
        self._load()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

class TransformerEncoderWrapper(nn.Module):
    """
    Wraps a HuggingFace AutoModel to expose only `last_hidden_state`.

    The HuggingFace model is loaded lazily on first forward pass so that
    the wrapper can be constructed without network access.

    Args:
        model_name:  HuggingFace model identifier.
        device:      Torch device string ("cpu" or "cuda").

    Example::

        encoder = TransformerEncoderWrapper("bert-base-uncased")
        hidden = encoder(input_ids, attention_mask)  # [B, T, 768]
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.auto_model: Optional[nn.Module] = None
        self._inference_mode: bool = False

    # ------------------------------------------------------------------
    def load_pretrained(self) -> None:
        """Load the HuggingFace backbone (called lazily on first forward)."""
        from transformers import AutoModel
        self.auto_model = AutoModel.from_pretrained(self.model_name)
        self.auto_model.to(self.device)

    # ------------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        """Return the hidden dimension of the loaded model."""
        if self.auto_model is None:
            self.load_pretrained()
        return self.auto_model.config.hidden_size

    # ------------------------------------------------------------------
    @contextmanager
    def inference_mode(self):
        """
        Context manager that switches the encoder to eval + no_grad.

        Usage::

            with encoder.inference_mode():
                embeddings = encoder(input_ids, attention_mask)
        """
        prev_training = self.training
        self.eval()
        with torch.no_grad():
            yield
        self.train(prev_training)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run the transformer backbone and return the last hidden state.

        Args:
            input_ids:      LongTensor [batch, seq_len]
            attention_mask: LongTensor [batch, seq_len]
            token_type_ids: LongTensor [batch, seq_len]  (optional)

        Returns:
            Tensor [batch, seq_len, hidden_dim]
        """
        if self.auto_model is None:
            self.load_pretrained()

        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.auto_model(**kwargs)
        return outputs.last_hidden_state
