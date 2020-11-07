import torch.nn as nn
import numpy as np
from einops import reduce

class BaseNet(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
    
    def forward(self, input):
        raise NotImplementedError

    def summary(self):
        net_parameters = [list(param.shape) for param in self.parameters() if param.requires_grad]
        total = reduce(np.array([reduce(np.array(p_sh), 'l ->', 'prod') for p_sh in net_parameters]), 'p ->', 'sum')
        print(f'net_parameters: {list(net_parameters)}\ntotal: {total}')
