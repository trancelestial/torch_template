import torch
from einops import reduce

def L1(y, pred, reduction='mean'):
    l1 = reduce(reduce(torch.abs(y - pred), 'b d -> b', 'sum'), 'b ->', reduction)

    return {'total': l1, 'debug': {'L1': l1}} # plotting

def L2(y, pred, reduction='mean'):
    l2 = reduce(reduce((y - pred)**2, 'b d -> b', 'sum'), 'b ->', reduction)
    return {'total': l2, 'debug': {'L2': l2}} # plotting
