import torch.nn as nn
from .pooling import MeanPooling, MaxPooling, CLSPooling

class SentenceEncoder(nn.Module):
    """Combines Transformer Encoder + Pooling layer to output fixed-size sentence vector."""
    def __init__(self, encoder, pooling_mode: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling_mode = pooling_mode.lower()
        if self.pooling_mode == "mean":
            self.pooling = MeanPooling()
        elif self.pooling_mode == "max":
            self.pooling = MaxPooling()
        elif self.pooling_mode == "cls":
            self.pooling = CLSPooling()
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        token_embeddings = self.encoder(input_ids, attention_mask, token_type_ids)
        sentence_embedding = self.pooling(token_embeddings, attention_mask)
        return sentence_embedding
