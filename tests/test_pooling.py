import torch
import pytest
from sbert_reproduction.models.pooling import MeanPooling, MaxPooling, CLSPooling

def test_mean_pooling_padding():
    pooling = MeanPooling()
    # Batch size 2, Seq len 3, Hidden dim 4
    token_embeddings = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [100.0, 100.0, 100.0, 100.0]],
        [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    ])
    # Sentence 1 has 2 real tokens; 3rd is padding. Sentence 2 has 1 real token.
    attention_mask = torch.tensor([
        [1, 1, 0],
        [1, 0, 0]
    ])

    pooled = pooling(token_embeddings, attention_mask)
    assert pooled.shape == (2, 4)

    # Expected sentence 0 mean of token 0 & 1: [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
    expected_0 = torch.tensor([3.0, 4.0, 5.0, 6.0])
    assert torch.allclose(pooled[0], expected_0)

    # Expected sentence 1 mean of token 0: [2, 4, 6, 8]
    expected_1 = torch.tensor([2.0, 4.0, 6.0, 8.0])
    assert torch.allclose(pooled[1], expected_1)

def test_max_pooling_padding():
    pooling = MaxPooling()
    token_embeddings = torch.tensor([
        [[1.0, 10.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [100.0, 100.0, 100.0, 100.0]]
    ])
    attention_mask = torch.tensor([[1, 1, 0]]) # 3rd token is padding (100.0 ignored)

    pooled = pooling(token_embeddings, attention_mask)
    assert pooled.shape == (1, 4)
    expected = torch.tensor([5.0, 10.0, 7.0, 8.0])
    assert torch.allclose(pooled[0], expected)

def test_cls_pooling():
    pooling = CLSPooling()
    token_embeddings = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    ])
    pooled = pooling(token_embeddings)
    assert pooled.shape == (1, 4)
    assert torch.allclose(pooled[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
