import torch
from sbert_reproduction.losses import SoftmaxLoss, CosineSimilarityLoss, TripletLoss

def test_softmax_loss_shape():
    loss_fn = SoftmaxLoss(sentence_embedding_dimension=4, num_labels=3, concatenation_mode="u_v_absdiff")
    u = torch.randn(2, 4)
    v = torch.randn(2, 4)
    labels = torch.tensor([0, 1])

    loss = loss_fn(u, v, labels)
    assert loss.dim() == 0 # scalar loss

def test_cosine_similarity_loss_shape():
    loss_fn = CosineSimilarityLoss()
    u = torch.randn(4, 8)
    v = torch.randn(4, 8)
    labels = torch.tensor([0.8, 0.2, 1.0, 0.0])

    loss = loss_fn(u, v, labels)
    assert loss.dim() == 0

def test_triplet_loss_margin():
    loss_fn = TripletLoss(margin=1.0)
    a = torch.tensor([[0.0, 0.0]])
    p = torch.tensor([[0.1, 0.0]]) # dist = 0.1
    n = torch.tensor([[2.0, 0.0]]) # dist = 2.0

    # dist_pos (0.1) - dist_neg (2.0) + margin (1.0) = -0.9 -> relu = 0.0
    loss = loss_fn(a, p, n)
    assert float(loss.item()) == 0.0
