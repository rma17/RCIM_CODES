# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:26:29 2022

@author: 97091
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
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,GNNExplainer
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import KFold

def full_index(l):
   # index1=[]
   # index2=[]
   # for x in range(6):
   #   for y in range(6):
   #      if(x==y):
   #          continue
   #      else:
   #          index1.append(x)
   #          index2.append(y)
   # index1=np.array(index1)
   # index2=np.array(index2)
   # index1=index1.reshape(1,30)
   # index2=index2.reshape(1,30)
   # index=np.concatenate((index1,index2),axis=0)
   if l==1:
    index=[[0,1,3,4],
           [3,4,0,1]]
   else:
     index=[[0,2,3,4],
            [3,4,0,2]] 
   # index=[[0,1,2,3,4],[4,3,2,1,0]]
   # index=[[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],
   #        [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]]
   # index=[[0,0,1,1,2,2],[1,2,0,2,0,1]]
   index=[[0,1,2,3],[1,2,3,4]]
   # index=[[0,1,1,2,2,3,3,4],
   #         [1,0,2,1,3,2,4,3]]
   # index=[[0,1,2,3,4],[4,3,2,1,0]]
   # index=[[0,2,3,4],
   #        [3,4,0,2]] 
   # index=[[0,1,2,3,4,5],[3,4,5,0,1,2]]
  
   return index


def process_csv(path,path1,path2,path3,path4):
    
    
    
    
     datas = pd.read_csv(path).values
     
     indi= pd.read_csv(path1).values
     
     label=pd.read_csv(path2).values
     state=pd.read_csv(path3).values
     goal=pd.read_csv(path4).values
     

     


     a={i:list(np.where(indi==i)[0]) for i in np.unique(indi)}
     BATCH=[]
     for k in a.keys():
      x=torch.tensor(datas[a[k],:],dtype=torch.float)
      
      edge=full_index(0)
      
      edge_index=torch.tensor(edge,dtype=torch.long)
      
      # l=torch.nn.functional.one_hot(torch.tensor(label[k]),num_classes=6)
      l=torch.tensor(label[k],dtype=torch.long)
     
      s=torch.tensor(state[k],dtype=torch.long)
      g=torch.tensor([goal[k]],dtype=torch.float)
      
      data=Data(x=x,edge_index=edge_index,y=l,state=s,goal=g)
      BATCH.append(data)
    
     return BATCH,state

    
    
    
    
     datas = pd.read_csv(path).values
     
     indi= pd.read_csv(path1).values
     
     label=pd.read_csv(path2).values
     state=pd.read_csv(path3).values
     goal=pd.read_csv(path4).values
     

     


     a={i:list(np.where(indi==i)[0]) for i in np.unique(indi)}
     BATCH=[]
     for k in a.keys():
      x=torch.tensor(datas[a[k],:],dtype=torch.float)
      
      
      edge=full_index(0)
      edge_index=torch.tensor(edge,dtype=torch.long)
      
      # l=torch.nn.functional.one_hot(torch.tensor(label[k]),num_classes=6)
      l=torch.tensor(label[k],dtype=torch.long)
     
      s=torch.tensor(state[k],dtype=torch.long)
      g=torch.tensor([goal[k]],dtype=torch.float)
      
      data=Data(x=x,edge_index=edge_index,y=l,state=s,goal=g)
      BATCH.append(data)
    
     return BATCH,state
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 =SAGEConv(4, hidden_channels) 
        self.conv2 =SAGEConv(hidden_channels, hidden_channels)  
        self.conv3 =SAGEConv(hidden_channels, hidden_channels) 
        self.lin = Linear(3+3, 3)
        self.importance=Linear(hidden_channels,6)
        # self.soft_max=torch.nn.softmax(dim=1)
        
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
        
        
            
           
        # im=torch.nn.functional.softmax(importance)
        im=torch.nn.functional.normalize(importance, p=2.0, dim=1, eps=1e-12, out=None)
        
        
        
        
        x1 = self.layer_1(torch.cat([abs(im),goal.float()],dim = 1))
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
    
    def test(self,x,edge_index,batch,goal):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        
        importance=self.importance(x)
        
        
            
           
            
        im=torch.nn.functional.normalize(importance, p=2.0, dim=1, eps=1e-12, out=None)
        
        
        
        
        x1 = self.layer_1(torch.cat([abs(im),goal.float()],dim = 1))
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
        print(abs(im),label.argmax(dim=1))
        return x 
dataset,state=process_csv("Gra.csv","I1.csv","S1.csv","L1.csv","G1.csv")

# import random
# random.shuffle(dataset)

model = GNN(hidden_channels=128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
criterion1=torch.nn.CrossEntropyLoss()


def test(loader):
    model.eval()
    correct=0
    correct1=0
    loader=DataLoader(loader, batch_size=8, shuffle=True)
    print(len(loader))
    for data in loader:
        out,label=model(data.x,data.edge_index,data.batch,data.goal)
        # print(out)
        pred1=out.argmax(dim=1)
        pred=label.argmax(dim=1)
        correct += int((pred == data.state).sum())
        correct1+=int((pred1==data.y).sum())
   
  
    return correct/len(loader.dataset),correct1/len(loader.dataset)
def test1(loader):
    model.eval()
    c=0
   
    loader=DataLoader(loader,batch_size=8)
    # print(len(loader))
    for data in loader:
        correct=0
        correct1=0
        out,label=model(data.x,data.edge_index,data.batch,data.goal)
        # print(out)
        pred1=out.argmax(dim=1)
        pred=label.argmax(dim=1)
        correct = int((pred == data.state).sum())
        correct1=int((pred1==data.y).sum())
        
        if correct==8 and correct1==8:
            c+=1
        
      
       
    return c
def test2(loader):
    model.eval()
    c=0
   
    loader=DataLoader(loader,batch_size=6)
    print(len(loader))
    for data in loader:
        correct=0
        correct1=0
        out,label=model(data.x,data.edge_index,data.batch,data.goal)
        # print(out)
        pred1=out.argmax(dim=1)
        pred=label.argmax(dim=1)
        correct = int((pred == data.state).sum())
        correct1=int((pred1==data.y).sum())
        
        if correct==6 and correct1==6:
            c+=1
        
      
       
    return c

kf = KFold(n_splits=5)
kf.get_n_splits(dataset)

def split(index,dataset):
    splited=[]
    for i in index:
        splited.append(dataset[i])
    return splited

def train_model(datasets):
    model.train()
    datas=DataLoader(datasets, batch_size=16, shuffle=True)
    for _ in range(500):
      for data in datas:
        out,label=model(data.x,data.edge_index,data.batch,data.goal)
       
        loss1=criterion1(out,data.y)
        loss2=criterion1(label,data.state)
        loss=loss1+loss2
       
        loss.backward()
     
        optimizer.step()
        optimizer.zero_grad()
        
        

for train_index,test_index in kf.split(dataset):
    train_data,test_data,train_state,test_state=split(train_index,dataset), split(test_index,dataset),split(train_index,state),split(test_index,state)
    
    train_model(train_data)
    acc,acc1=test(test_data)
    
    print(acc,acc1)
# torch.save(model.state_dict(), 'correction4.pt')    



model.eval()
dataset,state=process_csv("gra4.csv","I4.csv","S4.csv","L4.csv","G4.csv")
dataset1,state=process_csv("gra4t.csv","I44.csv","S44.csv","L44.csv","G44.csv")
dataset2,state=process_csv("Gra55.csv","I55.csv","S55.csv","L55.csv","G55.csv")
acc=test1(dataset)
acc2=test2(dataset2)
acc1=test1(dataset1)

print(acc)
x=torch.tensor([
[-0.175,-0.65,0.2,0],

[-0.576,  -0.5087095,  0.2,1],

[-0.175,	-0.6261,	0.2,0],

[-0.1756,	-0.5,	0.2,1],
[-0.06,	-0.65,	0.2,0]


])
edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]])
batch=torch.tensor([0]*x.size()[0])
a=model.test(x,edge_index,batch,dataset[2].goal)