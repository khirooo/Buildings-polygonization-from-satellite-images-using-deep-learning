import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomCRFLoss(nn.Module):
    def __init__(self, num_classes, spatial_weight=1.0, bilateral_weight=1.0, theta_alpha=1.0, theta_beta=1.0, theta_gamma=1.0):
        super(CustomCRFLoss, self).__init__()
        self.num_classes = num_classes
        self.spatial_weight = spatial_weight
        self.bilateral_weight = bilateral_weight
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma

    def forward(self, logits, labels, images):
        unary = self._get_unary_from_logits(logits, labels)
        pairwise = self._get_pairwise(images)
        return self._dense_crf(unary, pairwise, images)

    def _get_unary_from_logits(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        n, c, h, w = probs.shape
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, c)
        labels = labels.view(-1)
        loss = F.nll_loss(torch.log(probs), labels, reduction='none')
        return loss.view(n, h, w)

    def _get_pairwise(self, images):
        n, c, h, w = images.shape
        pairwise = torch.zeros(n, h, w, h, w, dtype=torch.float32).to(images.device)

        for i in range(h):
            for j in range(w):
                diff_i = (images[:, :, i, j].unsqueeze(2) - images[:, :, i, :]).pow(2).sum(1)
                diff_j = (images[:, :, i, j].unsqueeze(2) - images[:, :, :, j]).pow(2).sum(1)
                pairwise[:, i, j, :, :] = self.spatial_weight * torch.exp(-diff_i / (2 * self.theta_alpha ** 2)) + \
                                          self.bilateral_weight * torch.exp(-diff_j / (2 * self.theta_beta ** 2))

        return pairwise

    def _dense_crf(self, unary, pairwise, images):
        n, h, w = unary.shape
        pairwise = pairwise.view(n, h * w, h * w)
        unary = unary.view(n, h * w)
        
        Q = unary.clone()

        for _ in range(5):  # Number of iterations
            Q = Q - pairwise.bmm(Q.unsqueeze(2)).squeeze(2) / self.theta_gamma

        return torch.mean(Q)


