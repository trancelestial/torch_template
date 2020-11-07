import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.layers import *
from base.base_net import BaseNet


class FcNet(BaseNet):
    def __init__(self, size_in, size_out, layers=[], device='cuda', **kwargs):
        super().__init__(size_in, size_out)
        self.device=device
        
        fc_size_in = size_in
        self.fc = []
        for fc_size_out in layers:
            self.fc.append(torch.nn.Linear(fc_size_in, fc_size_out))
            fc_size_in = fc_size_out
        self.fc = torch.nn.ModuleList(self.fc)

        self.last = torch.nn.Linear(fc_size_in, size_out)
        
    def forward(self, input):
        x = input
        for fc in self.fc:
            x = fc(x)
            x = F.relu(x)
        x = self.last(x)

        return x
