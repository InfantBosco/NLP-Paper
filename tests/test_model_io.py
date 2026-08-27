"""
Model save/load and inference-mode tests — Stage 5 unit tests.

Covers:
- SBERTModel.save_pretrained / load_pretrained round-trip
- State dict consistency before and after save/load
- Inference-only mode (no_grad context, eval mode flag)
- ClassificationHead and RegressionHead save/load
"""

import os
import json
import tempfile

import pytest
import torch
import torch.nn as nn

from sbert_reproduction.models.pooling import MeanPooling
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import (
    SBERTModel,
    ClassificationHead,
    RegressionHead,
)


# ---------------------------------------------------------------------------
# Dummy encoder (no HuggingFace download required)
# ---------------------------------------------------------------------------

class DummyEncoder(nn.Module):
    """Deterministic encoder: returns a fixed random matrix of shape [B, T, H]."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)  # has learnable weights

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        B, T = input_ids.shape
        # Use input_ids as seeds for deterministic output
        x = torch.zeros(B, T, self.hidden_dim)
        return self.linear(x)


def make_sbert_model(hidden_dim=16, pooling_mode="mean") -> SBERTModel:
    encoder = DummyEncoder(hidden_dim=hidden_dim)
    sent_enc = SentenceEncoder(encoder, pooling_mode=pooling_mode)
    return SBERTModel(sent_enc)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_expected_files(self):
        model = make_sbert_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "encoder_weights.pt"))
            assert os.path.isfile(os.path.join(tmpdir, "model_config.json"))

    def test_model_config_json_is_valid(self):
        model = make_sbert_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            config_path = os.path.join(tmpdir, "model_config.json")
            with open(config_path) as fh:
                cfg = json.load(fh)
            assert "model_class" in cfg
            assert "pooling_mode" in cfg
            assert cfg["model_class"] == "SBERTModel"

    def test_load_pretrained_restores_weights(self):
        """Weights before save must equal weights after load."""
        model = make_sbert_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Build fresh model with same architecture
            encoder2  = DummyEncoder(hidden_dim=16)
            sent_enc2 = SentenceEncoder(encoder2, pooling_mode="mean")
            restored  = SBERTModel.load_pretrained(tmpdir, sent_enc2)

        # Compare all parameter tensors
        for (name, p_orig), (_, p_rest) in zip(
            model.named_parameters(), restored.named_parameters()
        ):
            assert torch.allclose(p_orig, p_rest), \
                f"Parameter {name} differs after load"

    def test_load_pretrained_returns_eval_mode(self):
        model = make_sbert_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            encoder2  = DummyEncoder(hidden_dim=16)
            sent_enc2 = SentenceEncoder(encoder2, pooling_mode="mean")
            restored  = SBERTModel.load_pretrained(tmpdir, sent_enc2)
        assert not restored.training, "Loaded model should be in eval mode"

    def test_load_missing_file_raises(self):
        encoder2  = DummyEncoder(hidden_dim=16)
        sent_enc2 = SentenceEncoder(encoder2, pooling_mode="mean")
        with pytest.raises(FileNotFoundError):
            SBERTModel.load_pretrained("/nonexistent/path", sent_enc2)

    def test_forward_output_consistent_after_load(self):
        """Same input must produce identical output before and after save/load."""
        torch.manual_seed(99)
        model = make_sbert_model()
        input_ids      = torch.zeros(2, 6, dtype=torch.long)
        attention_mask = torch.ones(2, 6, dtype=torch.long)

        fa = {"input_ids": input_ids, "attention_mask": attention_mask}
        fb = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.no_grad():
            u_orig, v_orig = model(fa, fb)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            encoder2  = DummyEncoder(hidden_dim=16)
            sent_enc2 = SentenceEncoder(encoder2, pooling_mode="mean")
            restored  = SBERTModel.load_pretrained(tmpdir, sent_enc2)

        with torch.no_grad():
            u_rest, v_rest = restored(fa, fb)

        assert torch.allclose(u_orig, u_rest, atol=1e-6)
        assert torch.allclose(v_orig, v_rest, atol=1e-6)


# ---------------------------------------------------------------------------
# Inference-only mode
# ---------------------------------------------------------------------------

class TestInferenceMode:

    def test_inference_mode_no_grad(self):
        """Inside inference_mode(), grad should not be tracked."""
        sent_enc = SentenceEncoder(DummyEncoder(16), pooling_mode="mean")
        input_ids      = torch.zeros(2, 5, dtype=torch.long)
        attention_mask = torch.ones(2, 5, dtype=torch.long)

        with sent_enc.inference_mode():
            out = sent_enc(input_ids, attention_mask)
        # No grad_fn means grad was not tracked
        assert out.grad_fn is None

    def test_inference_mode_sets_eval(self):
        sent_enc = SentenceEncoder(DummyEncoder(16), pooling_mode="mean")
        sent_enc.train()
        with sent_enc.inference_mode():
            assert not sent_enc.training
        # After context exits, training flag is restored
        assert sent_enc.training

    def test_inference_mode_restores_train(self):
        sent_enc = SentenceEncoder(DummyEncoder(16), pooling_mode="cls")
        sent_enc.train()
        with sent_enc.inference_mode():
            pass
        assert sent_enc.training

    def test_inference_mode_restores_eval(self):
        sent_enc = SentenceEncoder(DummyEncoder(16), pooling_mode="cls")
        sent_enc.eval()
        with sent_enc.inference_mode():
            pass
        assert not sent_enc.training


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------

class TestClassificationHead:

    @pytest.mark.parametrize("mode,expected_in", [
        ("u_v_absdiff", 3),
        ("u_v", 2),
        ("absdiff", 1),
        ("mult", 1),
        ("absdiff_mult", 2),
        ("u_v_mult", 3),
        ("u_v_absdiff_mult", 4),
    ])
    def test_input_size(self, mode, expected_in):
        dim = 8
        head = ClassificationHead(embedding_dim=dim, num_labels=3, concatenation_mode=mode)
        assert head.linear.in_features == dim * expected_in

    def test_output_shape_u_v_absdiff(self):
        head = ClassificationHead(embedding_dim=16, num_labels=3)
        u = torch.randn(4, 16)
        v = torch.randn(4, 16)
        logits = head(u, v)
        assert logits.shape == (4, 3)

    def test_output_num_labels(self):
        for n in [2, 3, 5]:
            head = ClassificationHead(embedding_dim=8, num_labels=n)
            u = torch.randn(2, 8)
            v = torch.randn(2, 8)
            assert head(u, v).shape == (2, n)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ClassificationHead(embedding_dim=8, num_labels=3, concatenation_mode="invalid")


# ---------------------------------------------------------------------------
# RegressionHead
# ---------------------------------------------------------------------------

class TestRegressionHead:

    def test_output_shape_no_labels(self):
        head = RegressionHead()
        u   = torch.randn(6, 32)
        v   = torch.randn(6, 32)
        out = head(u, v)
        assert out.shape == (6,)

    def test_output_range(self):
        """Cosine similarity must lie in [-1, 1]."""
        head = RegressionHead()
        torch.manual_seed(7)
        u   = torch.randn(20, 64)
        v   = torch.randn(20, 64)
        out = head(u, v)
        assert (out >= -1.0 - 1e-5).all()
        assert (out <=  1.0 + 1e-5).all()

    def test_returns_scalar_loss_with_labels(self):
        head   = RegressionHead()
        u      = torch.randn(4, 16)
        v      = torch.randn(4, 16)
        labels = torch.tensor([0.5, 0.8, 0.1, 0.9])
        loss   = head(u, v, labels)
        assert loss.dim() == 0, "Loss must be a scalar"
        assert float(loss) >= 0.0

    def test_identical_vectors_cosine_one(self):
        head = RegressionHead()
        v   = normalize_embeddings_helper(torch.randn(3, 8))
        out = head(v, v)
        assert torch.allclose(out, torch.ones(3), atol=1e-5)


def normalize_embeddings_helper(t):
    import torch.nn.functional as F
    return F.normalize(t, p=2, dim=-1)
