"""
Full SBERT model: Siamese network + classification / regression heads +
checkpoint save/load.

Implements the three-objective SBERT architecture from
Reimers & Gurevych (2019):

  A. Classification objective (Section 3.1):
       o = softmax( W_t · (u, v, |u − v|) )

  B. Regression objective (Section 3.2):
       loss = MSE(cos(u, v), label)

  C. Triplet objective (Section 3.3):
       loss = max(d(a, p) − d(a, n) + margin, 0)

Official code reference:
  sentence-transformers/sentence_transformers/  (independent re-implementation)
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Linear classification head used in the NLI / classification objective.

    Concatenates sentence embeddings u, v, and |u − v| (or any subset
    controlled by *concatenation_mode*), then projects to *num_labels*
    classes.

    Mathematical definition (paper Eq. 1):
        o = softmax( W_t · (u, v, |u − v|) )
        where  W_t ∈ ℝ^{k × 3d},  d = embedding_dim,  k = num_labels.

    Paper reference: Section 3.1.
    Configuration fields: ``num_labels``, ``concatenation_mode``.

    Args:
        embedding_dim:       Dimension of each sentence embedding (d).
        num_labels:          Number of output classes (k).
        concatenation_mode:  One of ``"u_v_absdiff"`` (default, from paper),
                             ``"u_v"``, ``"absdiff"``, ``"mult"``,
                             ``"u_v_absdiff_mult"``.
    """

    _INPUT_SIZES = {
        "u_v_absdiff":      3,
        "u_v":              2,
        "absdiff":          1,
        "mult":             1,
        "absdiff_mult":     2,
        "u_v_mult":         3,
        "u_v_absdiff_mult": 4,
    }

    def __init__(
        self,
        embedding_dim: int,
        num_labels: int = 3,
        concatenation_mode: str = "u_v_absdiff",
    ) -> None:
        super().__init__()
        if concatenation_mode not in self._INPUT_SIZES:
            raise ValueError(
                f"Unknown concatenation_mode '{concatenation_mode}'. "
                f"Valid options: {list(self._INPUT_SIZES)}"
            )
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.concatenation_mode = concatenation_mode
        in_features = embedding_dim * self._INPUT_SIZES[concatenation_mode]
        self.linear = nn.Linear(in_features, num_labels)

    def _build_features(self, u: Tensor, v: Tensor) -> Tensor:
        abs_diff = torch.abs(u - v)
        mult     = u * v
        mode     = self.concatenation_mode
        if mode == "u_v_absdiff":
            return torch.cat([u, v, abs_diff], dim=1)
        elif mode == "u_v":
            return torch.cat([u, v], dim=1)
        elif mode == "absdiff":
            return abs_diff
        elif mode == "mult":
            return mult
        elif mode == "absdiff_mult":
            return torch.cat([abs_diff, mult], dim=1)
        elif mode == "u_v_mult":
            return torch.cat([u, v, mult], dim=1)
        elif mode == "u_v_absdiff_mult":
            return torch.cat([u, v, abs_diff, mult], dim=1)
        raise AssertionError("unreachable")

    def forward(self, u: Tensor, v: Tensor) -> Tensor:
        """
        Args:
            u: Tensor [batch, embedding_dim]
            v: Tensor [batch, embedding_dim]

        Returns:
            logits Tensor [batch, num_labels]
        """
        features = self._build_features(u, v)
        return self.linear(features)

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_labels={self.num_labels}, "
            f"concatenation_mode={self.concatenation_mode!r}"
        )


# ---------------------------------------------------------------------------
# Regression Head
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """
    Cosine-similarity regression head for the STS objective.

    Computes cosine similarity between u and v and optionally scales
    it to the label range used in the dataset.

    Mathematical definition (paper Section 3.2):
        ŷ = cos(u, v)
        loss = MSE(ŷ, y)

    Score scaling:
        STS-B labels are in [0, 5].  The paper normalises them to [0, 1]
        before computing cosine similarity loss.  This class stores the
        original range so the caller can rescale predictions as needed.

    Paper reference: Section 3.2.
    Configuration fields: ``score_min``, ``score_max``.

    Args:
        score_min: Minimum label value (0 for STS-B after normalisation).
        score_max: Maximum label value (1 after normalisation, 5 raw).
    """

    def __init__(self, score_min: float = 0.0, score_max: float = 1.0) -> None:
        super().__init__()
        self.score_min = score_min
        self.score_max = score_max
        self.loss_fct  = nn.MSELoss()

    def forward(
        self,
        u: Tensor,
        v: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            u:      Tensor [batch, dim]
            v:      Tensor [batch, dim]
            labels: Tensor [batch]  — similarity scores (normalised to [-1, 1]
                    or [0, 1] depending on dataset).

        Returns:
            If *labels* is given: scalar loss Tensor.
            Otherwise:            cosine similarity Tensor [batch].
        """
        cos_sim = F.cosine_similarity(u, v, dim=-1)
        if labels is not None:
            return self.loss_fct(cos_sim, labels.view(-1))
        return cos_sim

    def extra_repr(self) -> str:
        return f"score_min={self.score_min}, score_max={self.score_max}"


# ---------------------------------------------------------------------------
# Main SBERT model
# ---------------------------------------------------------------------------

class SBERTModel(nn.Module):
    """
    Siamese network for sentence pair encoding.

    Holds a single :class:`SentenceEncoder` with **shared weights** for
    both sentence branches.  A classification or regression head can be
    attached separately; this class focuses on the encoder path only.

    Save / load
    -----------
    ``save_pretrained(directory)`` writes:
      - ``encoder_weights.pt``  — ``state_dict()`` of the full model
      - ``model_config.json``   — metadata (pooling mode, model name, etc.)

    ``SBERTModel.load_pretrained(directory, sentence_encoder)`` restores
    weights and returns the model ready for inference.

    Args:
        sentence_encoder: A :class:`SentenceEncoder` instance.
    """

    def __init__(self, sentence_encoder: nn.Module) -> None:
        super().__init__()
        self.sentence_encoder = sentence_encoder

    # ------------------------------------------------------------------
    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward a single sentence batch through the shared encoder."""
        return self.sentence_encoder(input_ids, attention_mask, token_type_ids)

    # ------------------------------------------------------------------
    def forward(
        self,
        features_a: dict,
        features_b: dict,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode a pair of sentence batches.

        Args:
            features_a: Dict with ``input_ids``, ``attention_mask``
                        (and optionally ``token_type_ids``).
            features_b: Same structure for sentence B.

        Returns:
            ``(u, v)`` — embeddings of sentence A and B,
            both Tensor [batch, hidden_dim].
        """
        u = self.encode(**features_a)
        v = self.encode(**features_b)
        return u, v

    # ------------------------------------------------------------------
    def encode_sentences(
        self,
        tokenizer,
        sentences: list,
        batch_size: int = 32,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        High-level inference helper: tokenise and encode raw sentences.

        Args:
            tokenizer:  :class:`TokenizerWrapper` instance.
            sentences:  List of raw strings.
            batch_size: Encoding batch size.
            normalize:  If True, L2-normalise output embeddings.

        Returns:
            Tensor [len(sentences), hidden_dim]  on CPU.
        """
        from .pooling import normalize_embeddings as _normalize

        embeddings = self.sentence_encoder.encode_text(
            tokenizer, sentences, batch_size=batch_size
        )
        if normalize:
            embeddings = _normalize(embeddings)
        return embeddings

    # ------------------------------------------------------------------
    def save_pretrained(self, directory: str) -> None:
        """
        Persist model weights and config to *directory*.

        Files created:
          - ``encoder_weights.pt``  — full ``state_dict()``
          - ``model_config.json``   — pooling mode, model name, version

        Args:
            directory: Path to output directory (created if absent).
        """
        os.makedirs(directory, exist_ok=True)

        # Save weights
        weights_path = os.path.join(directory, "encoder_weights.pt")
        torch.save(self.state_dict(), weights_path)

        # Save config metadata
        config: dict = {
            "model_class": self.__class__.__name__,
            "pooling_mode": getattr(self.sentence_encoder, "pooling_mode", "unknown"),
            "normalize": getattr(self.sentence_encoder, "normalize", False),
        }
        # Try to capture the underlying HuggingFace model name
        try:
            config["encoder_model_name"] = self.sentence_encoder.encoder.model_name
        except AttributeError:
            config["encoder_model_name"] = "unknown"

        config_path = os.path.join(directory, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    # ------------------------------------------------------------------
    @classmethod
    def load_pretrained(
        cls,
        directory: str,
        sentence_encoder: nn.Module,
        map_location: str = "cpu",
    ) -> "SBERTModel":
        """
        Restore a saved model from *directory*.

        Args:
            directory:        Path that was previously written by
                              :meth:`save_pretrained`.
            sentence_encoder: A freshly constructed :class:`SentenceEncoder`
                              whose architecture matches the saved weights.
            map_location:     Device to load weights onto.

        Returns:
            :class:`SBERTModel` with restored weights in eval mode.

        Raises:
            FileNotFoundError: If *directory* or the weights file is missing.
        """
        weights_path = os.path.join(directory, "encoder_weights.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {weights_path}"
            )

        model = cls(sentence_encoder)
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()
        return model
