import torch
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.pooling import MeanPooling

class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.hidden_dim)

def test_sentence_encoder_output_shape():
    encoder = SentenceEncoder(encoder=DummyEncoder(hidden_dim=16), pooling_mode="mean")
    input_ids = torch.ones((3, 10), dtype=torch.long)
    attention_mask = torch.ones((3, 10), dtype=torch.long)

    output = encoder(input_ids, attention_mask)
    assert output.shape == (3, 16)
