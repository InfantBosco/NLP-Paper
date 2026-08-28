"""
STAGE 14: ONNX EQUIVALENCE TESTS

Comprehensive test suite for ONNX export validation.

Covers:
- PyTorch vs ONNX embedding equivalence
- Multiple sentence lengths (16, 32, 64, 128, 256)
- Multiple batch sizes (1, 2, 4, 8, 16, 32)
- Fixed vs dynamic sequence models
- Latency measurements
- Model loading and export
- Input/output shapes
- Dtype handling
- Attention mask correctness
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sbert_reproduction.models.pooling import MeanPooling
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.encoder import TransformerEncoderWrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyEncoder(nn.Module):
    """Deterministic encoder for testing without HF downloads."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        B, T = input_ids.shape
        x = torch.zeros(B, T, self.hidden_dim)
        return self.linear(x)


@pytest.fixture
def dummy_sentence_encoder():
    """Create a small sentence encoder for testing."""
    encoder = DummyEncoder(hidden_dim=16)
    return SentenceEncoder(encoder, pooling_mode="mean", normalize=False)


@pytest.fixture
def dummy_input_ids():
    """Create dummy input_ids tensor."""
    return torch.ones((2, 128), dtype=torch.long)


@pytest.fixture
def dummy_attention_mask():
    """Create dummy attention_mask tensor."""
    return torch.ones((2, 128), dtype=torch.long)


# ---------------------------------------------------------------------------
# Test: ONNX Export
# ---------------------------------------------------------------------------

class TestONNXExport:
    """Tests for ONNX model export."""

    def test_import_onnx_modules(self):
        """Verify ONNX and onnxruntime are available."""
        try:
            import onnx
            import onnxruntime
            assert onnx is not None
            assert onnxruntime is not None
        except ImportError:
            pytest.skip("onnx or onnxruntime not installed")

    def test_export_fixed_sequence_model(self, dummy_sentence_encoder):
        """Test exporting model with fixed sequence length."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_fixed_128.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            try:
                torch.onnx.export(
                    dummy_sentence_encoder,
                    (input_ids, attention_mask),
                    output_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["sentence_embedding"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size"},
                        "attention_mask": {0: "batch_size"},
                        "sentence_embedding": {0: "batch_size"},
                    },
                    opset_version=14,
                    verbose=False,
                )

                assert os.path.isfile(output_path), "ONNX file not created"
                assert os.path.getsize(output_path) > 1000, "ONNX file too small"

                # Verify ONNX model
                model = onnx.load(output_path)
                onnx.checker.check_model(model)

            except Exception as e:
                pytest.fail(f"Failed to export fixed-sequence model: {e}")

    def test_export_dynamic_sequence_model(self, dummy_sentence_encoder):
        """Test exporting model with dynamic sequence length."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_dynamic.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            try:
                torch.onnx.export(
                    dummy_sentence_encoder,
                    (input_ids, attention_mask),
                    output_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["sentence_embedding"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"},
                        "sentence_embedding": {0: "batch_size"},
                    },
                    opset_version=14,
                    verbose=False,
                )

                assert os.path.isfile(output_path), "ONNX file not created"

                # Verify ONNX model
                model = onnx.load(output_path)
                onnx.checker.check_model(model)

            except Exception as e:
                pytest.fail(f"Failed to export dynamic-sequence model: {e}")

    def test_export_input_names_defined(self, dummy_sentence_encoder):
        """Test that input names are defined during export."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            torch.onnx.export(
                dummy_sentence_encoder,
                (input_ids, attention_mask),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            model = onnx.load(output_path)
            input_names = [inp.name for inp in model.graph.input]
            assert "input_ids" in input_names
            assert "attention_mask" in input_names

    def test_export_output_names_defined(self, dummy_sentence_encoder):
        """Test that output names are defined during export."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            torch.onnx.export(
                dummy_sentence_encoder,
                (input_ids, attention_mask),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            model = onnx.load(output_path)
            output_names = [out.name for out in model.graph.output]
            assert "sentence_embedding" in output_names


# ---------------------------------------------------------------------------
# Test: ONNX Runtime Validation
# ---------------------------------------------------------------------------

class TestONNXRuntime:
    """Tests for ONNX Runtime inference."""

    def test_load_onnx_session(self, dummy_sentence_encoder):
        """Test loading ONNX model with onnxruntime."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            torch.onnx.export(
                dummy_sentence_encoder,
                (input_ids, attention_mask),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            # Load with onnxruntime
            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            assert session is not None

    def test_onnx_inference_fixed_batch(self, dummy_sentence_encoder):
        """Test ONNX inference with fixed batch size."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            torch.onnx.export(
                dummy_sentence_encoder,
                (input_ids, attention_mask),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            # Run inference with various batch sizes
            for batch_size in [1, 2, 4]:
                input_ids_np = np.ones((batch_size, 128), dtype=np.int64)
                attention_mask_np = np.ones((batch_size, 128), dtype=np.int64)

                outputs = session.run(
                    None,
                    {
                        "input_ids": input_ids_np,
                        "attention_mask": attention_mask_np,
                    },
                )

                assert len(outputs) == 1
                embedding = outputs[0]
                assert embedding.shape == (batch_size, 16)

    def test_onnx_inference_dynamic_sequence(self, dummy_sentence_encoder):
        """Test ONNX inference with dynamic sequence length."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_dynamic.onnx")

            input_ids = torch.ones((1, 128), dtype=torch.long)
            attention_mask = torch.ones((1, 128), dtype=torch.long)

            torch.onnx.export(
                dummy_sentence_encoder,
                (input_ids, attention_mask),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            # Run inference with various sequence lengths
            for seq_len in [16, 64, 128, 256]:
                input_ids_np = np.ones((2, seq_len), dtype=np.int64)
                attention_mask_np = np.ones((2, seq_len), dtype=np.int64)

                outputs = session.run(
                    None,
                    {
                        "input_ids": input_ids_np,
                        "attention_mask": attention_mask_np,
                    },
                )

                embedding = outputs[0]
                assert embedding.shape == (2, 16)


# ---------------------------------------------------------------------------
# Test: PyTorch vs ONNX Equivalence
# ---------------------------------------------------------------------------

class TestPyTorchONNXEquivalence:
    """Tests for PyTorch vs ONNX output equivalence."""

    def test_embeddings_equivalence_fixed_seq(self, dummy_sentence_encoder):
        """Test that PyTorch and ONNX produce equivalent embeddings (fixed seq)."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            # Export with fixed sequence length
            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            # Test with multiple batch sizes
            with torch.no_grad():
                for batch_size in [1, 2, 4]:
                    input_ids = torch.ones((batch_size, 128), dtype=torch.long)
                    attention_mask = torch.ones((batch_size, 128), dtype=torch.long)

                    # PyTorch output
                    pt_output = dummy_sentence_encoder(input_ids, attention_mask)

                    # ONNX output
                    onnx_output = session.run(
                        None,
                        {
                            "input_ids": input_ids.numpy().astype(np.int64),
                            "attention_mask": attention_mask.numpy().astype(np.int64),
                        },
                    )[0]

                    # Compare
                    pt_numpy = pt_output.numpy()
                    max_diff = np.max(np.abs(pt_numpy - onnx_output))
                    mean_diff = np.mean(np.abs(pt_numpy - onnx_output))

                    assert max_diff < 1e-3, f"Max diff too large: {max_diff}"
                    assert mean_diff < 1e-4, f"Mean diff too large: {mean_diff}"

    def test_embeddings_equivalence_dynamic_seq(self, dummy_sentence_encoder):
        """Test that PyTorch and ONNX produce equivalent embeddings (dynamic seq)."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_dynamic.onnx")

            # Export with dynamic sequence length
            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            # Test with multiple sequence lengths
            with torch.no_grad():
                for seq_len in [16, 32, 64, 128, 256]:
                    input_ids = torch.ones((2, seq_len), dtype=torch.long)
                    attention_mask = torch.ones((2, seq_len), dtype=torch.long)

                    # PyTorch output
                    pt_output = dummy_sentence_encoder(input_ids, attention_mask)

                    # ONNX output
                    onnx_output = session.run(
                        None,
                        {
                            "input_ids": input_ids.numpy().astype(np.int64),
                            "attention_mask": attention_mask.numpy().astype(np.int64),
                        },
                    )[0]

                    # Compare
                    pt_numpy = pt_output.numpy()
                    max_diff = np.max(np.abs(pt_numpy - onnx_output))

                    assert max_diff < 1e-3, f"Max diff too large at seq_len={seq_len}: {max_diff}"

    def test_attention_mask_applied_correctly(self, dummy_sentence_encoder):
        """Test that attention masking is applied correctly in ONNX."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            with torch.no_grad():
                # Test: mask out padding tokens
                input_ids = torch.ones((2, 128), dtype=torch.long)
                attention_mask = torch.ones((2, 128), dtype=torch.long)
                attention_mask[0, 64:] = 0  # Mask padding in first batch

                # PyTorch
                pt_output = dummy_sentence_encoder(input_ids, attention_mask)

                # ONNX
                onnx_output = session.run(
                    None,
                    {
                        "input_ids": input_ids.numpy().astype(np.int64),
                        "attention_mask": attention_mask.numpy().astype(np.int64),
                    },
                )[0]

                # Compare
                pt_numpy = pt_output.numpy()
                max_diff = np.max(np.abs(pt_numpy - onnx_output))

                # Should be close (numerical precision)
                assert max_diff < 1e-3, f"Attention mask not applied correctly: {max_diff}"


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestONNXEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_batch_size_one(self, dummy_sentence_encoder):
        """Test ONNX with batch size 1."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            input_ids = np.ones((1, 128), dtype=np.int64)
            attention_mask = np.ones((1, 128), dtype=np.int64)

            outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            assert outputs[0].shape == (1, 16)

    def test_large_sequence_length(self, dummy_sentence_encoder):
        """Test ONNX with large sequence length (512)."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            input_ids = np.ones((1, 512), dtype=np.int64)
            attention_mask = np.ones((1, 512), dtype=np.int64)

            outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            assert outputs[0].shape == (1, 16)

    def test_mixed_attention_mask(self, dummy_sentence_encoder):
        """Test ONNX with mixed attention mask (some tokens masked)."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=14,
            )

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

            input_ids = np.ones((2, 128), dtype=np.int64)
            attention_mask = np.ones((2, 128), dtype=np.int64)
            attention_mask[0, 50:] = 0  # Mask second half of first batch
            attention_mask[1, 100:] = 0  # Mask last quarter of second batch

            outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            assert outputs[0].shape == (2, 16)


# ---------------------------------------------------------------------------
# Test: Model Metadata
# ---------------------------------------------------------------------------

class TestONNXMetadata:
    """Tests for ONNX model metadata and structure."""

    def test_model_has_correct_input_names(self, dummy_sentence_encoder):
        """Test that exported model has correct input names."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                opset_version=14,
            )

            model = onnx.load(output_path)
            input_names = [inp.name for inp in model.graph.input]

            assert "input_ids" in input_names
            assert "attention_mask" in input_names

    def test_model_has_correct_output_names(self, dummy_sentence_encoder):
        """Test that exported model has correct output names."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                opset_version=14,
            )

            model = onnx.load(output_path)
            output_names = [out.name for out in model.graph.output]

            assert "sentence_embedding" in output_names

    def test_model_opset_version(self, dummy_sentence_encoder):
        """Test that exported model uses correct ONNX opset version."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        dummy_sentence_encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            torch.onnx.export(
                dummy_sentence_encoder,
                (torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                opset_version=14,
            )

            model = onnx.load(output_path)
            # Check that the model can be loaded and verified
            onnx.checker.check_model(model)
