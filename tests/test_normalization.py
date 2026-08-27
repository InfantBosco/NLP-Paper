"""
Normalization and cosine similarity tests — Stage 5 unit tests.

Covers:
- normalize_embeddings: unit norms, zero vector, batch shapes
- cosine_similarity: identical vectors = 1, orthogonal = 0, anti-parallel = -1
- cosine_similarity_matrix: shape, symmetry, diagonal = 1 after normalization
"""

import pytest
import torch
from sbert_reproduction.models.pooling import normalize_embeddings, cosine_similarity
from sbert_reproduction.models.similarity import cosine_similarity_matrix


class TestNormalizeEmbeddings:

    def test_output_shape_unchanged(self):
        emb = torch.randn(8, 32)
        out = normalize_embeddings(emb)
        assert out.shape == emb.shape

    def test_unit_norms(self):
        """Every output row must have L2-norm ≈ 1."""
        torch.manual_seed(0)
        emb  = torch.randn(16, 64)
        out  = normalize_embeddings(emb)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-6), \
            f"Max norm deviation: {(norms - 1).abs().max()}"

    def test_zero_vector_does_not_crash(self):
        """The zero vector has undefined direction; eps must prevent NaN."""
        emb = torch.zeros(3, 8)
        out = normalize_embeddings(emb)
        assert not torch.isnan(out).any(), "NaN produced for zero vector"
        assert out.shape == (3, 8)

    def test_already_normalized_unchanged(self):
        """Applying normalization to an already-unit vector should be idempotent."""
        emb  = torch.randn(4, 16)
        emb  = normalize_embeddings(emb)
        emb2 = normalize_embeddings(emb)
        assert torch.allclose(emb, emb2, atol=1e-6)

    def test_single_vector(self):
        emb = torch.tensor([[3.0, 4.0]])   # norm = 5
        out = normalize_embeddings(emb)
        expected = torch.tensor([[0.6, 0.8]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_batch_size_one(self):
        emb = torch.randn(1, 128)
        out = normalize_embeddings(emb)
        assert abs(float(out.norm(p=2, dim=-1)) - 1.0) < 1e-6


class TestCosineSimilarity:

    def test_identical_vectors_return_one(self):
        torch.manual_seed(1)
        v   = torch.randn(4, 32)
        sim = cosine_similarity(v, v)
        assert torch.allclose(sim, torch.ones(4), atol=1e-5), f"Got {sim}"

    def test_orthogonal_vectors_return_zero(self):
        """e1 = [1, 0] and e2 = [0, 1] are orthogonal."""
        u   = torch.tensor([[1.0, 0.0]])
        v   = torch.tensor([[0.0, 1.0]])
        sim = cosine_similarity(u, v)
        assert torch.allclose(sim, torch.zeros(1), atol=1e-6), f"Got {sim}"

    def test_anti_parallel_vectors_return_minus_one(self):
        u   = torch.tensor([[1.0, 0.0, 0.0]])
        v   = torch.tensor([[-1.0, 0.0, 0.0]])
        sim = cosine_similarity(u, v)
        assert torch.allclose(sim, torch.tensor([-1.0]), atol=1e-5), f"Got {sim}"

    def test_output_range(self):
        """cosine similarity must lie in [-1, 1]."""
        torch.manual_seed(2)
        u   = torch.randn(32, 64)
        v   = torch.randn(32, 64)
        sim = cosine_similarity(u, v)
        assert (sim >= -1.0 - 1e-5).all() and (sim <= 1.0 + 1e-5).all()

    def test_output_shape(self):
        u   = torch.randn(8, 16)
        v   = torch.randn(8, 16)
        sim = cosine_similarity(u, v)
        assert sim.shape == (8,)

    def test_symmetry(self):
        """cos(u, v) == cos(v, u)."""
        torch.manual_seed(3)
        u   = torch.randn(5, 12)
        v   = torch.randn(5, 12)
        assert torch.allclose(cosine_similarity(u, v), cosine_similarity(v, u), atol=1e-6)


class TestCosineSimilarityMatrix:

    def test_output_shape(self):
        a = torch.randn(5, 32)
        b = torch.randn(7, 32)
        mat = cosine_similarity_matrix(a, b)
        assert mat.shape == (5, 7)

    def test_diagonal_is_one_for_same_matrix(self):
        """S[i, i] must equal 1 when a == b (identical vectors)."""
        torch.manual_seed(4)
        a   = torch.randn(6, 16)
        mat = cosine_similarity_matrix(a, a)
        diag = mat.diag()
        assert torch.allclose(diag, torch.ones(6), atol=1e-5), f"Diag: {diag}"

    def test_values_in_range(self):
        torch.manual_seed(5)
        a   = torch.randn(10, 64)
        b   = torch.randn(10, 64)
        mat = cosine_similarity_matrix(a, b)
        assert mat.min() >= -1.0 - 1e-5
        assert mat.max() <=  1.0 + 1e-5

    def test_symmetric_when_a_equals_b(self):
        torch.manual_seed(6)
        a   = torch.randn(8, 32)
        mat = cosine_similarity_matrix(a, a)
        assert torch.allclose(mat, mat.t(), atol=1e-5)

    def test_square_and_nonsquare(self):
        a = torch.randn(3, 8)
        b = torch.randn(5, 8)
        assert cosine_similarity_matrix(a, b).shape == (3, 5)
        assert cosine_similarity_matrix(b, a).shape == (5, 3)
