import torch.nn as nn

class SBERTModel(nn.Module):
    """Siamese Network wrapper holding a single SentenceEncoder with shared tied weights."""
    def __init__(self, sentence_encoder):
        super().__init__()
        self.sentence_encoder = sentence_encoder

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        return self.sentence_encoder(input_ids, attention_mask, token_type_ids)

    def forward(self, features_a, features_b):
        u = self.encode(**features_a)
        v = self.encode(**features_b)
        return u, v
