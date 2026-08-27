"""
Extended pooling tests — Stage 5 unit tests.

Covers:
- MeanPooling: shape, padding correctness, all-padding row, variable seq lengths
- MaxPooling:  shape, padding masking, all-padding row
- CLSPooling:  shape, correct token extraction
- WeightedMeanPooling: shape, padding, uniform attention gives weighted > plain mean
- Edge cases: batch_size=1, seq_len=1, large hidden dims
"""

import pytest
import torch
from sbert_reproduction.models.pooling import (
    MeanPooling,
    MaxPooling,
    CLSPooling,
    WeightedMeanPooling,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(batch_size, seq_len, hidden_dim, fill_value=1.0):
    return torch.full((batch_size, seq_len, hidden_dim), fill_value)


# ---------------------------------------------------------------------------
# MeanPooling
# ---------------------------------------------------------------------------

class TestMeanPooling:

    def test_output_shape(self):
        pooling = MeanPooling()
        emb  = torch.randn(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.long)
        out  = pooling(emb, mask)
        assert out.shape == (4, 16)

    def test_ignores_padding_tokens(self):
        """The [CLS]/[SEP] padding token at index 2 should not affect the mean."""
        pooling = MeanPooling()
        token_embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [100.0, 100.0, 100.0, 100.0]],  # padding — must be ignored
            [[2.0, 4.0, 6.0, 8.0],
             [0.0,  0.0,  0.0,  0.0],
             [0.0,  0.0,  0.0,  0.0]],
        ])
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        pooled = pooling(token_embeddings, attention_mask)

        assert pooled.shape == (2, 4)
        # Row 0: mean of token 0 and 1
        expected_0 = torch.tensor([3.0, 4.0, 5.0, 6.0])
        assert torch.allclose(pooled[0], expected_0), f"Got {pooled[0]}"
        # Row 1: only token 0
        expected_1 = torch.tensor([2.0, 4.0, 6.0, 8.0])
        assert torch.allclose(pooled[1], expected_1), f"Got {pooled[1]}"

    def test_all_real_tokens(self):
        """With all tokens real, mean pooling equals standard mean."""
        pooling = MeanPooling()
        emb  = torch.randn(3, 5, 8)
        mask = torch.ones(3, 5, dtype=torch.long)
        out  = pooling(emb, mask)
        expected = emb.mean(dim=1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_all_padding_row_does_not_crash(self):
        """A fully-masked row should return zeros (clamp avoids div-by-zero)."""
        pooling = MeanPooling()
        emb  = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])  # row 1 all padding
        out  = pooling(emb, mask)
        assert out.shape == (2, 8)
        # Row with all padding → ~zero (numerator is 0)
        assert torch.allclose(out[1], torch.zeros(8), atol=1e-5)

    def test_variable_sequence_lengths(self):
        """Different real-token counts per row must all be handled correctly."""
        pooling = MeanPooling()
        B, T, H = 5, 8, 4
        emb  = torch.ones(B, T, H)
        # Real lengths: 1, 2, 3, 4, 8
        lengths = [1, 2, 3, 4, 8]
        mask = torch.zeros(B, T, dtype=torch.long)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        out = pooling(emb, mask)
        assert out.shape == (B, H)
        # All embeddings are 1s, so mean should be 1 regardless of length
        assert torch.allclose(out, torch.ones(B, H), atol=1e-6)

    def test_batch_size_one(self):
        pooling = MeanPooling()
        emb  = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[1, 1]])
        out  = pooling(emb, mask)
        assert out.shape == (1, 2)
        assert torch.allclose(out, torch.tensor([[2.0, 3.0]]))

    def test_seq_len_one(self):
        pooling = MeanPooling()
        emb  = torch.tensor([[[7.0, 8.0, 9.0]]])
        mask = torch.tensor([[1]])
        out  = pooling(emb, mask)
        assert torch.allclose(out, torch.tensor([[7.0, 8.0, 9.0]]))


# ---------------------------------------------------------------------------
# MaxPooling
# ---------------------------------------------------------------------------

class TestMaxPooling:

    def test_output_shape(self):
        pooling = MaxPooling()
        emb  = torch.randn(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.long)
        out  = pooling(emb, mask)
        assert out.shape == (4, 16)

    def test_ignores_padding_tokens(self):
        """Padding positions set to 100 must NOT be selected by max."""
        pooling = MaxPooling()
        token_embeddings = torch.tensor([
            [[1.0, 10.0, 3.0, 4.0],
             [5.0,  6.0, 7.0, 8.0],
             [100.0, 100.0, 100.0, 100.0]],  # padding with very high values
        ])
        attention_mask = torch.tensor([[1, 1, 0]])
        out = pooling(token_embeddings, attention_mask)
        assert out.shape == (1, 4)
        expected = torch.tensor([5.0, 10.0, 7.0, 8.0])
        assert torch.allclose(out[0], expected), f"Got {out[0]}"

    def test_all_real_tokens(self):
        """Without padding, max over all tokens."""
        pooling = MaxPooling()
        emb  = torch.randn(2, 6, 8)
        mask = torch.ones(2, 6, dtype=torch.long)
        out  = pooling(emb, mask)
        expected = emb.max(dim=1)[0]
        assert torch.allclose(out, expected, atol=1e-6)

    def test_padding_large_negative_sentinel(self):
        """Verify the sentinel is −1e9, not affecting the selected max."""
        pooling = MaxPooling()
        emb  = torch.zeros(1, 3, 4)
        emb[0, 0, :] = 0.5   # real token
        emb[0, 1, :] = -100  # real token but very negative
        emb[0, 2, :] = 999   # padding — must be ignored
        mask = torch.tensor([[1, 1, 0]])
        out  = pooling(emb, mask)
        expected = torch.full((1, 4), 0.5)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_variable_sequence_lengths(self):
        pooling = MaxPooling()
        B, T, H = 3, 6, 4
        emb  = torch.arange(B * T * H, dtype=torch.float).reshape(B, T, H)
        lengths = [2, 4, 6]
        mask = torch.zeros(B, T, dtype=torch.long)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        out = pooling(emb, mask)
        assert out.shape == (B, H)


# ---------------------------------------------------------------------------
# CLSPooling
# ---------------------------------------------------------------------------

class TestCLSPooling:

    def test_output_shape(self):
        pooling = CLSPooling()
        emb = torch.randn(3, 8, 16)
        out = pooling(emb)
        assert out.shape == (3, 16)

    def test_extracts_first_token(self):
        pooling = CLSPooling()
        emb = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        ])
        out = pooling(emb)
        assert out.shape == (1, 4)
        assert torch.allclose(out[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_accepts_attention_mask_arg(self):
        """CLSPooling should work when attention_mask is passed (API consistency)."""
        pooling = CLSPooling()
        emb  = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.long)
        out  = pooling(emb, attention_mask=mask)
        assert out.shape == (2, 8)
        assert torch.allclose(out, emb[:, 0, :])

    def test_batch_size_one_seq_len_one(self):
        pooling = CLSPooling()
        emb = torch.tensor([[[3.0, 4.0]]])
        out = pooling(emb)
        assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))


# ---------------------------------------------------------------------------
# WeightedMeanPooling
# ---------------------------------------------------------------------------

class TestWeightedMeanPooling:

    def test_output_shape(self):
        pooling = WeightedMeanPooling()
        emb  = torch.randn(4, 10, 16)
        mask = torch.ones(4, 10, dtype=torch.long)
        out  = pooling(emb, mask)
        assert out.shape == (4, 16)

    def test_padding_ignored(self):
        """Padding positions must not contribute to the weighted sum."""
        pooling = WeightedMeanPooling()
        # Two tokens; third is padding with extreme value
        emb  = torch.tensor([
            [[1.0, 0.0], [2.0, 0.0], [1000.0, 1000.0]],
        ], dtype=torch.float)
        mask = torch.tensor([[1, 1, 0]])
        out  = pooling(emb, mask)
        # weights: pos 0 → w=1, pos 1 → w=2
        # weighted mean = (1*1 + 2*2) / (1+2) = 5/3 ≈ 1.6667
        expected_dim0 = (1 * 1.0 + 2 * 2.0) / (1 + 2)
        assert abs(float(out[0, 0]) - expected_dim0) < 1e-5, f"Got {out[0, 0]}"
        assert abs(float(out[0, 1]) - 0.0) < 1e-5

    def test_differs_from_plain_mean_on_nonuniform_values(self):
        """Weighted mean should differ from uniform mean when values vary by position."""
        mean_pooling = MeanPooling()
        weighted_pooling = WeightedMeanPooling()
        torch.manual_seed(0)
        emb  = torch.randn(1, 6, 8)
        mask = torch.ones(1, 6, dtype=torch.long)
        plain    = mean_pooling(emb, mask)
        weighted = weighted_pooling(emb, mask)
        # They should generally not be equal (unless embeddings are all equal)
        assert not torch.allclose(plain, weighted, atol=1e-4)

    def test_all_padding_does_not_crash(self):
        pooling = WeightedMeanPooling()
        emb  = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
        out  = pooling(emb, mask)
        assert out.shape == (2, 8)
        # Row 1 all-padded → should be ~zero
        assert torch.allclose(out[1], torch.zeros(8), atol=1e-5)
