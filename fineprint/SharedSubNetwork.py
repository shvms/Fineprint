from torch import nn
from torch.nn import functional as F

# TODO: Set correct axis on required methods as per the input shape

class SharedSubNetwork(nn.Module):
  def __init__(self, n_bits: int):
    super(SharedSubNetwork, self).__init__()
    self.conv11 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
    self.conv12 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    self.conv21 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2)
    self.conv22 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    self.conv31 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)
    self.conv32 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    self.conv41 = nn.Conv2d(in_channels=384, out_channels=1024, kernel_size=3, stride=1)
    self.conv42 = nn.Conv2d(in_channels=1024, out_channels=50*n_bits, kernel_size=1, stride=1)
    self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1)
  
  def forward(self, x):
    x = F.relu(self.conv11(x))
    x = F.relu(self.conv12(x))
    x = self.maxpool1(x)
    
    x = F.relu(self.conv21(x))
    x = F.relu(self.conv22(x))
    x = self.maxpool2(x)
    
    x = F.relu(self.conv31(x))
    x = F.relu(self.conv32(x))
    x = self.maxpool3(x)
    
    x = F.relu(self.conv41(x))
    x = F.relu(self.conv42(x))
    
    return self.avgpool(x)

if __name__ == '__main__':
  subnetwork = SharedSubNetwork(10)
  print(subnetwork)
