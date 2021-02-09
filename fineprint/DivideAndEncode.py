import torch
from torch import nn
from torch.nn import functional as F
import copy

class SingleModule(nn.Module):
  def __init__(self):
    super(SingleModule, self).__init__()
    self.linear = nn.Linear(in_features=50, out_features=1, bias=False)
  
  def forward(self, x):
    return F.sigmoid(self.linear(x))


class DivideAndEncode(nn.Module):
  def __init__(self, n_bits: int):
    super(DivideAndEncode, self).__init__()
    self.n_bits = n_bits
    self.modules = nn.ModuleList([copy.deepcopy(SingleModule()) for _ in range(n_bits)])
  
  def forward(self, conv_vector):
    splits = torch.split(conv_vector, 50, dim=1)
    output = []
    
    for idx, split in enumerate(splits):
      output.append(self.modules[idx].forward(split))
    
    return torch.Tensor(output)
