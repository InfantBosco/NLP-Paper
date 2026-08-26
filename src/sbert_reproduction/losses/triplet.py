import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TripletLoss(nn.Module):
    """
    Triplet Loss for Anchor (a), Positive (p), Negative (n).
    Loss = max(||s_a - s_p|| - ||s_a - s_n|| + margin, 0)
    Default margin = 1.0 (matching paper Section 3).
    """
    def __init__(self, margin: float = 1.0, distance_metric: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric.lower()

    def forward(self, rep_anchor: Tensor, rep_pos: Tensor, rep_neg: Tensor) -> Tensor:
        if self.distance_metric == "euclidean":
            dist_pos = F.pairwise_distance(rep_anchor, rep_pos, p=2)
            dist_neg = F.pairwise_distance(rep_anchor, rep_neg, p=2)
        elif self.distance_metric == "cosine":
            dist_pos = 1 - F.cosine_similarity(rep_anchor, rep_pos)
            dist_neg = 1 - F.cosine_similarity(rep_anchor, rep_neg)
        elif self.distance_metric == "manhattan":
            dist_pos = F.pairwise_distance(rep_anchor, rep_pos, p=1)
            dist_neg = F.pairwise_distance(rep_anchor, rep_neg, p=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()
