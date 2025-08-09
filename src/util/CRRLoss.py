import torch
from pykeen.losses import Loss

class CRRLoss(Loss):
    def __init__(self, t=1.0, p=0.1):
        super().__init__()
        self.t = t  # Temperature (tau)
        self.p = p  # Pressure (rho)

    def cliff_sigmoid(self, x):
        return 1 / (1 + torch.exp((self.p - x) / self.t))

    def forward(self, scores, labels):
        # scores: (batch_size, num_negatives + 1) - includes positive and negative scores
        # labels: (batch_size, num_negatives + 1) - 1 for positive, 0 for negative
        pos_scores = scores[labels == 1]  # Extract positive scores
        neg_scores = scores[labels == 0]  # Extract negative scores
        # Reshape for broadcasting: (batch_size, 1) - (batch_size, num_negatives)
        diff = neg_scores - pos_scores.unsqueeze(-1)
        return torch.log(torch.sum(self.cliff_sigmoid(diff), dim=1) + 1.0).mean()