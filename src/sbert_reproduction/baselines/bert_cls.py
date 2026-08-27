"""
Vanilla BERT with CLS-token pooling baseline.

Un-finetuned bert-base-uncased. The [CLS] token representation at
position 0 of the last hidden state is used as the sentence embedding.

Paper reference: Section 5, Table 2 — "BERT without fine-tuning (CLS)".
Configuration fields: model_name, max_seq_length.
Known differences: No task-specific fine-tuning.
"""

from __future__ import annotations

from ._bert_baseline_mixin import _BERTBaselineMixin
from sbert_reproduction.models.encoder import TransformerEncoderWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder


class VanillaBERTCLSBaseline(_BERTBaselineMixin):
    """
    Un-finetuned BERT with CLS-token pooling.

    Args:
        model_name:      HuggingFace model identifier.
        max_seq_length:  Maximum token length (truncation).
    """

    pooling_mode = "cls"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_seq_length: int = 128,
    ) -> None:
        self.model_name     = model_name
        self.max_seq_length = max_seq_length
        encoder             = TransformerEncoderWrapper(model_name)
        self.model          = SentenceEncoder(encoder, pooling_mode="cls")
