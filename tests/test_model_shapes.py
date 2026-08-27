"""
Model shape tests — Stage 5 unit tests.

Covers:
- SentenceEncoder output shapes for all four pooling modes
- SBERTModel encode() output shape
- ClassificationHead input/output sizes
- RegressionHead output sizes
- WeightedMeanPooling inside SentenceEncoder
"""

import pytest
import torch
import torch.nn as nn

from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import (
    SBERTModel,
    ClassificationHead,
    RegressionHead,
)


# ---------------------------------------------------------------------------
# Dummy encoder (no HuggingFace download)
# ---------------------------------------------------------------------------

class DummyEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        B, T = input_ids.shape
        return torch.randn(B, T, self.hidden_dim)


# ---------------------------------------------------------------------------
# SentenceEncoder — all pooling modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pooling_mode,hidden_dim", [
    ("mean",        16),
    ("max",         16),
    ("cls",         16),
    ("weightedmean", 32),
])
def test_sentence_encoder_output_shape(pooling_mode, hidden_dim):
    encoder = SentenceEncoder(
        encoder=DummyEncoder(hidden_dim=hidden_dim),
        pooling_mode=pooling_mode,
    )
    B, T = 3, 10
    input_ids      = torch.ones((B, T), dtype=torch.long)
    attention_mask = torch.ones((B, T), dtype=torch.long)

    output = encoder(input_ids, attention_mask)
    assert output.shape == (B, hidden_dim), \
        f"[{pooling_mode}] Expected ({B}, {hidden_dim}), got {output.shape}"


def test_sentence_encoder_invalid_pooling_raises():
    with pytest.raises(ValueError, match="Unknown pooling_mode"):
        SentenceEncoder(DummyEncoder(), pooling_mode="invalid_mode")


# ---------------------------------------------------------------------------
# SentenceEncoder with normalize=True
# ---------------------------------------------------------------------------

def test_sentence_encoder_normalized_unit_norms():
    """When normalize=True, output embeddings must have unit L2-norm."""
    encoder = SentenceEncoder(
        encoder=DummyEncoder(hidden_dim=16),
        pooling_mode="mean",
        normalize=True,
    )
    ids  = torch.ones((4, 8), dtype=torch.long)
    mask = torch.ones((4, 8), dtype=torch.long)
    out  = encoder(ids, mask)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-6), \
        f"Norms: {norms}"


# ---------------------------------------------------------------------------
# SBERTModel encode()
# ---------------------------------------------------------------------------

def test_sbert_model_encode_shape():
    se    = SentenceEncoder(DummyEncoder(hidden_dim=16), pooling_mode="mean")
    model = SBERTModel(se)
    ids   = torch.ones((5, 12), dtype=torch.long)
    mask  = torch.ones((5, 12), dtype=torch.long)
    out   = model.encode(ids, mask)
    assert out.shape == (5, 16)


def test_sbert_model_forward_returns_two_tensors():
    se    = SentenceEncoder(DummyEncoder(hidden_dim=16), pooling_mode="mean")
    model = SBERTModel(se)
    B, T  = 4, 8
    fa    = {"input_ids": torch.ones(B, T, dtype=torch.long),
             "attention_mask": torch.ones(B, T, dtype=torch.long)}
    fb    = {"input_ids": torch.ones(B, T, dtype=torch.long),
             "attention_mask": torch.ones(B, T, dtype=torch.long)}
    u, v  = model(fa, fb)
    assert u.shape == (B, 16)
    assert v.shape == (B, 16)


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,factor", [
    ("u_v_absdiff", 3),
    ("u_v", 2),
    ("absdiff", 1),
])
def test_classification_head_linear_in_features(mode, factor):
    dim  = 12
    head = ClassificationHead(embedding_dim=dim, num_labels=3, concatenation_mode=mode)
    assert head.linear.in_features == dim * factor


def test_classification_head_output_shape():
    head   = ClassificationHead(embedding_dim=16, num_labels=3)
    u      = torch.randn(8, 16)
    v      = torch.randn(8, 16)
    logits = head(u, v)
    assert logits.shape == (8, 3)


# ---------------------------------------------------------------------------
# RegressionHead
# ---------------------------------------------------------------------------

def test_regression_head_cosine_output_shape():
    head = RegressionHead()
    u    = torch.randn(6, 32)
    v    = torch.randn(6, 32)
    out  = head(u, v)
    assert out.shape == (6,)


def test_regression_head_loss_is_scalar():
    head   = RegressionHead()
    u      = torch.randn(4, 16)
    v      = torch.randn(4, 16)
    labels = torch.tensor([0.5, 0.8, 0.1, 0.9])
    loss   = head(u, v, labels)
    assert loss.dim() == 0
