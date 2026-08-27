"""
Vanilla BERT with attention-mask-aware mean pooling baseline.

Un-finetuned bert-base-uncased. All non-padding token representations
from the last hidden state are averaged to produce the sentence embedding.

Paper reference: Section 5, Table 2 — "BERT without fine-tuning (mean)".
Configuration fields: model_name, max_seq_length.
Known differences: No task-specific fine-tuning.
"""

from __future__ import annotations

from ._bert_baseline_mixin import _BERTBaselineMixin
from sbert_reproduction.models.encoder import TransformerEncoderWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder


class VanillaBERTMeanBaseline(_BERTBaselineMixin):
    """
    Un-finetuned BERT with attention-mask-aware mean pooling.

    Args:
        model_name:      HuggingFace model identifier.
        max_seq_length:  Maximum token length (truncation).
    """

    pooling_mode = "mean"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_seq_length: int = 128,
    ) -> None:
        self.model_name     = model_name
        self.max_seq_length = max_seq_length
        encoder             = TransformerEncoderWrapper(model_name)
        self.model          = SentenceEncoder(encoder, pooling_mode="mean")
