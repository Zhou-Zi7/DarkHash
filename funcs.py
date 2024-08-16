import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)

def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)

def algorithm2(P,Q,Y):
    eps = 0.0000001
    mean = P.mean(dim=1)
    mask1 = P>=mean
    mask2 = Y == Y.t()
    mask = mask1 == mask2
    loss =torch.mean(P * torch.log((P + eps) / (Q + eps)))
    return loss

class TPLoss(nn.Module):
    def __init__(self):
        super(TPLoss, self).__init__()
        self.eps = 0.0000001
    def forward(self, output_net, target_net):
        (n, d) = output_net.shape
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + self.eps)
        output_net[output_net != output_net] = 0
        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + self.eps)
        target_net[target_net != target_net] = 0
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        model_distance = 1 - model_similarity
        model_distance[range(n), range(n)] = 3
        model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
        model_distance[range(n), range(n)] = 0
        model_similarity = 1 - model_distance
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
        target_distance = 1 - target_similarity
        target_distance[range(n), range(n)] = 3
        target_distance = target_distance - torch.min(target_distance, dim=1)[0].view(-1, 1)
        target_distance[range(n), range(n)] = 0
        target_similarity = 1 - target_distance
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
        loss = CE(target_similarity, model_similarity)
        return loss
