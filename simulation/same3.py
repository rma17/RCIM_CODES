# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:24:45 2022

@author: 97091

This the GNN high-level decesion module
"""

import torch
# from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from torch_geometric.nn import GraphConv
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import ReLU
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GATConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import KFold

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 =SAGEConv(4, hidden_channels) 
        self.conv2 =SAGEConv(hidden_channels, hidden_channels)  
        self.conv3 =SAGEConv(hidden_channels, hidden_channels) 
        self.lin = Linear(3+3, 3)
        self.importance=Linear(hidden_channels,6)
        
        
        self.layer_1 = Linear(9, 512)
        self.layer_2 = Linear(512, 128)
        self.layer_3 = Linear(128, 64)
        self.layer_out = Linear(64, 5) 
        
        self.relu = ReLU()
        self.dropout =Dropout(p=0.2)
        self.batchnorm1 = BatchNorm1d(512)
        self.batchnorm2 = BatchNorm1d(128)
        self.batchnorm3 =BatchNorm1d(64)
        

    def forward(self, x, edge_index, batch,goal):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        
        importance=self.importance(x)
        im=torch.nn.functional.normalize(importance, p=2.0, dim=1, eps=1e-12, out=None)
        
       
        g=goal[self.selected-1]
        
        
        x1 = self.layer_1(torch.cat([im,g.float()],dim = 1))
        x1 = self.batchnorm1(x1)
        x1 = self.relu(x1)
        
        x1 = self.layer_2(x1)
        x1 = self.batchnorm2(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        x1 = self.layer_3(x1)
        x1 = self.batchnorm3(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        label = self.layer_out(x1)
        
        
        
        
        
        
        return im,label
    
    def select(self,x,edge_index,batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        importance=self.importance(x)
   
        im=torch.nn.functional.normalize(importance, p=2.0, dim=1, eps=1e-12, out=None)
        self.selected=im.argmax(dim=1)
        
        
        
       
        return x 
