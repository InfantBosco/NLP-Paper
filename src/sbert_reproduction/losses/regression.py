import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Regression Loss.
    Computes cosine similarity between sentence vectors u and v and minimizes MSE against normalized score y.
    """
    def __init__(self, loss_fct=nn.MSELoss()):
        super().__init__()
        self.loss_fct = loss_fct

    def forward(self, u: Tensor, v: Tensor, labels: Tensor):
        cos_sim = F.cosine_similarity(u, v)
        return self.loss_fct(cos_sim, labels.view(-1))
