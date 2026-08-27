"""
Batch encoding and variable sequence length tests — Stage 5 unit tests.

Covers:
- Batch size invariance (result must not depend on whether sentences are
  encoded together or one by one)
- Variable sequence lengths within a batch
- pairwise_encode shape and output
- batch_encode raises on empty input
- SBERTModel.forward shape across multiple batch sizes
- SentenceEncoder output with all three pooling modes across batch sizes
"""

import pytest
import torch
import torch.nn as nn

from sbert_reproduction.models.pooling import MeanPooling, MaxPooling, CLSPooling
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import SBERTModel
from sbert_reproduction.models.similarity import (
    cosine_similarity_matrix,
    pairwise_encode,
)


# ---------------------------------------------------------------------------
# Dummy encoder (no network download)
# ---------------------------------------------------------------------------

class LinearDummyEncoder(nn.Module):
    """Projects input_ids (treated as float) through a linear layer."""

    def __init__(self, vocab_size: int = 50, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.embed(input_ids)  # [B, T, H]


def make_encoder(hidden_dim=16, pooling_mode="mean"):
    backbone = LinearDummyEncoder(vocab_size=50, hidden_dim=hidden_dim)
    return SentenceEncoder(backbone, pooling_mode=pooling_mode)


# ---------------------------------------------------------------------------
# Batch-size invariance
# ---------------------------------------------------------------------------

class TestBatchSizeInvariance:

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_output_shape_across_batch_sizes(self, batch_size):
        """[batch_size, hidden_dim] output for all batch sizes."""
        enc  = make_encoder(hidden_dim=16, pooling_mode="mean")
        ids  = torch.randint(0, 50, (batch_size, 10))
        mask = torch.ones(batch_size, 10, dtype=torch.long)
        out  = enc(ids, mask)
        assert out.shape == (batch_size, 16), \
            f"Expected ({batch_size}, 16), got {out.shape}"

    def test_batch_split_gives_same_result(self):
        """
        Encoding sentences as a batch vs one-by-one must produce identical
        outputs (tests that no global batch-norm etc. is involved).
        """
        torch.manual_seed(10)
        enc  = make_encoder(hidden_dim=16, pooling_mode="mean")
        enc.eval()

        ids_batch  = torch.randint(0, 50, (4, 8))
        mask_batch = torch.ones(4, 8, dtype=torch.long)

        with torch.no_grad():
            batch_out = enc(ids_batch, mask_batch)  # [4, 16]
            single_outs = torch.stack([
                enc(ids_batch[i:i+1], mask_batch[i:i+1]).squeeze(0)
                for i in range(4)
            ])  # [4, 16]

        assert torch.allclose(batch_out, single_outs, atol=1e-6), \
            "Batch output differs from individual outputs"


# ---------------------------------------------------------------------------
# Variable sequence lengths
# ---------------------------------------------------------------------------

class TestVariableSequenceLengths:

    def test_shorter_real_lengths_same_shape(self):
        """Sentences with different real lengths (via attention_mask) must
        produce the same output shape."""
        enc  = make_encoder(hidden_dim=8, pooling_mode="mean")
        B, T = 4, 12
        ids  = torch.randint(0, 50, (B, T))
        # Different real lengths per sentence
        lengths = [3, 6, 9, 12]
        mask = torch.zeros(B, T, dtype=torch.long)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        out = enc(ids, mask)
        assert out.shape == (B, 8)

    @pytest.mark.parametrize("pooling_mode", ["mean", "max", "cls"])
    def test_all_pooling_modes_handle_variable_lengths(self, pooling_mode):
        enc  = make_encoder(hidden_dim=8, pooling_mode=pooling_mode)
        B, T = 3, 10
        ids  = torch.randint(0, 50, (B, T))
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        out = enc(ids, mask)
        assert out.shape == (B, 8)

    def test_sequence_length_one(self):
        """Degenerate case: every sequence has exactly 1 real token."""
        enc  = make_encoder(hidden_dim=8, pooling_mode="mean")
        B, T = 5, 1
        ids  = torch.randint(0, 50, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        out  = enc(ids, mask)
        assert out.shape == (B, 8)


# ---------------------------------------------------------------------------
# pairwise_encode
# ---------------------------------------------------------------------------

class TestPairwiseEncode:

    def test_output_shapes(self):
        enc = make_encoder(hidden_dim=16, pooling_mode="mean")
        B, T = 6, 8
        fa = {
            "input_ids":      torch.randint(0, 50, (B, T)),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
        fb = {
            "input_ids":      torch.randint(0, 50, (B, T)),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
        u, v = pairwise_encode(enc, fa, fb)
        assert u.shape == (B, 16)
        assert v.shape == (B, 16)

    def test_shared_weights(self):
        """Both u and v come from the same encoder — identical input → identical output."""
        enc = make_encoder(hidden_dim=16, pooling_mode="mean")
        enc.eval()
        B, T = 3, 6
        fa = {
            "input_ids":      torch.randint(0, 50, (B, T)),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
        with torch.no_grad():
            u, v = pairwise_encode(enc, fa, fa)
        assert torch.allclose(u, v, atol=1e-6), \
            "Same input through shared encoder must give same output"


# ---------------------------------------------------------------------------
# SBERTModel.forward
# ---------------------------------------------------------------------------

class TestSBERTModelForward:

    @pytest.mark.parametrize("batch_size", [1, 3, 7])
    def test_forward_output_shapes(self, batch_size):
        model = SBERTModel(make_encoder(hidden_dim=12, pooling_mode="mean"))
        B, T = batch_size, 8
        fa = {
            "input_ids":      torch.randint(0, 50, (B, T)),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
        fb = {
            "input_ids":      torch.randint(0, 50, (B, T)),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
        u, v = model(fa, fb)
        assert u.shape == (B, 12)
        assert v.shape == (B, 12)

    def test_forward_is_deterministic(self):
        """Same input, same seed → same output."""
        model = SBERTModel(make_encoder(hidden_dim=8, pooling_mode="mean"))
        model.eval()
        B, T = 2, 5
        fa = {
            "input_ids":      torch.randint(0, 50, (B, T)),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }
        with torch.no_grad():
            u1, v1 = model(fa, fa)
            u2, v2 = model(fa, fa)
        assert torch.allclose(u1, u2, atol=1e-6)
        assert torch.allclose(v1, v2, atol=1e-6)


# ---------------------------------------------------------------------------
# Empty / malformed inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_texts_raises_in_batch_encode(self):
        from sbert_reproduction.models.similarity import batch_encode

        class FakeTok:
            def tokenize(self, texts):
                return {
                    "input_ids": torch.zeros(len(texts), 8, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
                }

        enc = make_encoder(hidden_dim=8)
        with pytest.raises(ValueError, match="non-empty"):
            batch_encode(enc, FakeTok(), [])

    def test_encode_text_empty_raises(self):
        enc = make_encoder(hidden_dim=8)

        class FakeTok:
            def tokenize(self, texts):
                return {
                    "input_ids": torch.zeros(len(texts), 8, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
                }

        with pytest.raises(ValueError, match="non-empty"):
            enc.encode_text(FakeTok(), [])
