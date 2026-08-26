import torch
from sbert_reproduction.seed import set_seed

def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.randn(5, 5)
    set_seed(42)
    b = torch.randn(5, 5)
    assert torch.allclose(a, b)
