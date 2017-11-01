from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.autograd import Variable

class SDMLTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(SDMLTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.num_instances = 4

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = Variable(torch.FloatTensor(1).zero_()).cuda()
        num_class = n/self.num_instances
        num_positive_pairs = n * (self.num_instances-1)

	
        sum_dist_positive_all = Variable(torch.Tensor(1).zero_()).cuda()
        for i in range(n):
            sum_dist_positive_one = torch.sum(dist[i][mask[i]])
            sum_dist_positive_all += sum_dist_positive_one
        dist_mean = sum_dist_positive_all/num_positive_pairs
	

	    dist_ap, dist_an, loss_list = [], [], []
        for i in range(n):
	        dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
	        dist_i_sum = torch.sum(dist[i][mask[i]])
	        mask_i = mask[i].cpu().data
	        idx_of_j = torch.squeeze(torch.nonzero(mask_i)).numpy()
            for j in idx_of_j:
		        if j != i:
                    # Compute local loss
                    dist_ij = dist[i][j]
                    beta = (float(1)/num_class)*(dist_ij/dist_i_sum)
                    dist_ik_sum = torch.sum(torch.exp(self.margin-(dist[i][mask[i] == 0])))
                    dist_jl_sum = torch.sum(torch.exp(self.margin-(dist[j][mask[j] == 0])))
                    triplet_ij = torch.log(dist_ik_sum + dist_jl_sum) + dist_ij
                    loss_ij = beta * torch.pow(torch.clamp(triplet_ij, min=0.0, max=100), 2)
                    # Compute global loss
                    lambda0 = 0.005
                    loss_global = (lambda0/num_positive_pairs)*torch.pow(dist_ij-dist_mean, 2)
                    loss += (loss_ij + loss_global)
		            # loss += loss_ij

	    
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
