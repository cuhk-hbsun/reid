from __future__ import absolute_import
import sys
import torch
from torch import nn
from torch.autograd import Variable

class MSMLTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(MSMLTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
	    # For each batch, find the hardest positive and hardest negative
        dist_ap_max = torch.max(dist_ap)
        dist_an_min = torch.min(dist_an)
        # Compute ranking hinge loss
        y = dist_ap_max.data.new()
        y.resize_as_(dist_ap_max.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an_min, dist_ap_max, y)
        # loss = torch.log(1+torch.exp(dist_ap_min + self.margin - dist_an_max))
        prec = (dist_an.data > dist_ap.data).sum() * 1. / dist_an.size(0)
        return loss, prec


def range_except_k(k, end, start = 0):
    return range(start, k) + range(k+1, end)
	
