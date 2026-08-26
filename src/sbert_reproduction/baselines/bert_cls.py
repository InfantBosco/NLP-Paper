import torch
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.encoder import TransformerEncoderWrapper

class VanillaBERTCLSBaseline:
    """Un-finetuned pretrained BERT with CLS token pooling."""
    def __init__(self, model_name: str = "bert-base-uncased"):
        encoder = TransformerEncoderWrapper(model_name)
        self.model = SentenceEncoder(encoder, pooling_mode="cls")

    def encode(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
