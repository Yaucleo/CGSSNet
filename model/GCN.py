import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from GNN import GraphNetwork
import os
# device = torch.device("cuda")
# 是否使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1

# 定义图网络模型
class Graph_Conv_Network(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Graph_Conv_Network, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, (3, 3, 3), padding=1)
        self.BN = nn.BatchNorm3d(out_channels)
        self.active = nn.ReLU(inplace=True)

        self.GNN = GraphNetwork(in_channels, hidden_channels, out_channels).to(device)

    def forward(self, input):
        x = self.conv1(input)
        x = self.BN(x)
        x = self.active(x)
        x = x.permute(0, 4, 2, 3, 1)
        x = self.GNN(x)
        x = self.active(x)
        x = x.permute(0, 4, 2, 3, 1)
        x = x + input
        return x

