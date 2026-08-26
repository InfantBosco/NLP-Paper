import torch
import torch.nn as nn
from torch import Tensor

class MeanPooling(nn.Module):
    """Attention-mask aware Mean Pooling over token sequence."""
    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        # token_embeddings: [batch_size, seq_len, hidden_dim]
        # attention_mask:   [batch_size, seq_len]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

class MaxPooling(nn.Module):
    """Attention-mask aware Max Pooling over token sequence."""
    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Copy token embeddings and mask padding tokens with large negative value (-1e9)
        token_embeddings_masked = token_embeddings.clone()
        token_embeddings_masked[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings_masked, dim=1)[0]

class CLSPooling(nn.Module):
    """CLS Token Pooling (extracts the first token embedding vector)."""
    def forward(self, token_embeddings: Tensor, attention_mask: Tensor = None) -> Tensor:
        return token_embeddings[:, 0, :]
