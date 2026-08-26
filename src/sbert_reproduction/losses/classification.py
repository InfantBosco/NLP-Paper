import torch
import torch.nn as nn
from torch import Tensor

class SoftmaxLoss(nn.Module):
    """
    3-way NLI Classification Loss.
    Concatenates (u, v, |u - v|) into a linear classifier (3d -> num_labels) and computes Cross-Entropy.
    Paper Eq. 1: o = softmax(W_t (u, v, |u - v|))
    """
    def __init__(self, sentence_embedding_dimension: int, num_labels: int = 3, concatenation_mode: str = "u_v_absdiff"):
        super().__init__()
        self.embedding_dim = sentence_embedding_dimension
        self.num_labels = num_labels
        self.concatenation_mode = concatenation_mode

        if concatenation_mode == "u_v_absdiff":
            classifier_input_dim = 3 * sentence_embedding_dimension
        elif concatenation_mode == "u_v":
            classifier_input_dim = 2 * sentence_embedding_dimension
        elif concatenation_mode == "absdiff":
            classifier_input_dim = sentence_embedding_dimension
        elif concatenation_mode == "mult":
            classifier_input_dim = sentence_embedding_dimension
        elif concatenation_mode == "absdiff_mult":
            classifier_input_dim = 2 * sentence_embedding_dimension
        elif concatenation_mode == "u_v_mult":
            classifier_input_dim = 3 * sentence_embedding_dimension
        elif concatenation_mode == "u_v_absdiff_mult":
            classifier_input_dim = 4 * sentence_embedding_dimension
        else:
            raise ValueError(f"Unknown concatenation mode: {concatenation_mode}")

        self.classifier = nn.Linear(classifier_input_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, u: Tensor, v: Tensor, labels: Tensor = None):
        abs_diff = torch.abs(u - v)
        mult = u * v

        if self.concatenation_mode == "u_v_absdiff":
            features = torch.cat([u, v, abs_diff], dim=1)
        elif self.concatenation_mode == "u_v":
            features = torch.cat([u, v], dim=1)
        elif self.concatenation_mode == "absdiff":
            features = abs_diff
        elif self.concatenation_mode == "mult":
            features = mult
        elif self.concatenation_mode == "absdiff_mult":
            features = torch.cat([abs_diff, mult], dim=1)
        elif self.concatenation_mode == "u_v_mult":
            features = torch.cat([u, v, mult], dim=1)
        elif self.concatenation_mode == "u_v_absdiff_mult":
            features = torch.cat([u, v, abs_diff, mult], dim=1)

        logits = self.classifier(features)

        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits
