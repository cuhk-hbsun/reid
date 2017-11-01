from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
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
        dist_ap, dist_an, dist_bc = [], [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
	        mask_a = mask[i] == 1
	        mask_a = mask_a.data.cpu()
	        idx_a = torch.nonzero(mask_a)
	        idx_a = torch.squeeze(idx_a)
	        min_bc = Variable(torch.Tensor([1000])).cuda()
            for k in range(n):
		        if k not in idx_a:
		            min_bc_tmp = dist[k][mask[k] == 0].min()
		            min_bc = torch.min(min_bc, min_bc_tmp)
                    dist_bc.append(min_bc)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_bc = torch.cat(dist_bc)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss0 = self.ranking_loss(dist_an, dist_ap, y)
        loss1 = self.ranking_loss(dist_bc, dist_ap, y)
        loss = 0.8*loss0 + 0.2*loss1
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


def range_except_k(k, end, start = 0):
    return range(start, k) + range(k+1, end)
	
