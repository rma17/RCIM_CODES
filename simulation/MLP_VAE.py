# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:45:55 2021

@author: mrd
This the main function for task-aware subgoal planner
"""
import numpy as np
from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
import torch

from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader,TensorDataset
import pandas as pd
from sklearn.manifold import MDS

import numpy as np
from sklearn.model_selection import KFold
class UnitLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UnitLinear, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim).uniform_(0,1))
        self.bias = nn.Parameter(torch.zeros((out_dim,)))
        
    def forward(self, x):
        eps=nn.Parameter(self.weight/((1e-07)+torch.sqrt(torch.sum(torch.square(self.weight),
                                                  dim=0,
                                                  keepdims=True))))
        
        self.weight=eps
       
        return torch.matmul(x, self.weight) + self.bias

class Encoder(nn.Module):
    def __init__(self,num_feature,latent_dim,num_labels):
        super().__init__()
        ##Process data
        self.layer_1 = nn.Linear(num_feature,128)
        self.layer_2 = nn.Linear(128, 32)
        # self.layer_3 = nn.Linear(128, 32)
        ##Regressor layer
        self.layer_4=nn.Linear(32+num_labels,3)
        self.layer_5=nn.Linear(32+num_labels,3)
        #latent layer
        self.layer_6 = nn.Linear(32+num_labels, latent_dim)
        self.layer_7=nn.Linear(32+num_labels, latent_dim)
        
        # self.relu = nn.ReLU()
        self.tan=nn.Tanh()
       
        
        #output layer
        self.out_layer_regressor=UnitLinear(3+num_labels,latent_dim)
        self.out_layer_latent=nn.Linear(latent_dim+num_labels,latent_dim)
       
    def regressor(self,x,label):
        r_mean=self.layer_4(torch.cat([x,label.float()],dim = 1))
        r_log_var=self.layer_5(torch.cat([x,label.float()],dim = 1))
        return r_mean,r_log_var
    
    
    
    def latent_layer(self,x,label):
        z_mean=self.layer_6(torch.cat([x,label.float()],dim = 1))
        z_log_var=self.layer_7(torch.cat([x,label.float()],dim = 1))
        return z_mean,z_log_var
    
    
    def resampling(self,mu,log_var):
        
        
   
        batch = mu.shape[0]
        dim = mu.shape[1]
    # by default, random_normal has mean=0 and std=1.0
        eps = torch.randn(batch, dim).cuda()
        # std = torch.exp(0.5*log_var)
        # eps = torch.randn_like(std)
        # mu + torch.exp(0.5 * log_var) * eps
        return mu + torch.exp(0.5 * log_var) * eps
        
        # return z
    def forward(self,x,label):
       
        
        x = self.layer_1(x)
       
        x=self.tan(x)
        x=self.layer_2(x)
        x=self.tan(x)
        
        
        # x=self.layer_3(x)
        
        r_mean,r_log_var=self.regressor(x,label)
        
        z_mean,z_log_var=self.latent_layer(x,label)
       
        
        r=self.resampling(r_mean,r_log_var)
        
        z=self.resampling(z_mean, z_log_var)
     
        z=self.out_layer_latent(torch.cat([z,label],dim = 1))
        pz_mean=self.out_layer_regressor(torch.cat([r,label.float()],dim=1))
        # print(r)
        # print(pz_mean)
        
        return z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean
class Decoder(nn.Module):
    def __init__(self,latent_dim,out_dim):
        super().__init__()
        self.layer_1=nn.Linear(latent_dim,32)
        self.layer_2=nn.Linear(32,128)
        self.layer_3=nn.Linear(128,out_dim)
        self.tan=nn.Tanh()
    def forward(self,x):
        x=self.layer_1(x)
        x=self.tan(x)
        x=self.layer_2(x)
        x=self.tan(x)
        out=self.layer_3(x)
        
        return out
    
class VAE_Linear(nn.Module):
    def __init__(self,num_feature,latent_dim,out_dim,num_labels):
        super().__init__()
        
        self.encoder=Encoder(num_feature,latent_dim,num_labels)
        self.decoder=Decoder(latent_dim,out_dim)
        
        # self.mseLoss=nn.MSELoss()
   
    
    def forward(self,x,label):
        z_mean, z_log_var, latent, r_mean, r_log_var, r, pz_mean=self.encoder(x,label)
        out=self.decoder(latent)
        
        # print(out)
        return out,z_mean, z_log_var, r_mean, r_log_var, r, pz_mean
    
 
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()    
 
class VAE_Regression():
   def __init__(self,model,device):
       self.device=device
       self.model=model.to(self.device)
   def fit(self,batch,loss_func,optim,x,y,epoch,label):
         # self.model.apply(reset_weights)
         x=torch.tensor(x).to(self.device)
         y=torch.tensor(y).to(self.device)
         label=label.to(self.device)
         Dataset1=TensorDataset(x,y,label)
         trainloader=DataLoader(dataset=Dataset1,batch_size=batch,shuffle=True)
         for i in range(epoch):
             for x_1,y_1,label_1 in trainloader:
                 optim.zero_grad()
                 outputs=self.model(x_1.float(),label_1.float())
       
                 loss=loss_func( x_1,outputs,y_1,label_1.float())
                 # print(loss)
                 loss.backward()
                 optim.step()
   def test(self,X,Y,labels):
        X=torch.tensor(X).to(self.device)
        labels=labels.to(self.device)
        out,z_mean, z_log_var, r_mean, r_log_var, r, pz_mean=self.model(X.float(),labels.float())
        pred=r_mean.cpu()
        pred=pred.detach().numpy()
        m=mean_squared_error(Y, pred)
        r=r2_score(Y, pred)
        print("Mean squared error: %.3f" % mean_squared_error(Y, pred))
        print('R2 Variance score: %.3f' % r2_score(Y, pred))
        out=out.cpu().detach().numpy()
        return m,r,out
def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2
     
def loss_func(x,outputs,inputs_r,label):
        
        out,z_mean,z_log_var,r_mean,r_log_var,r,pz_mean=outputs
        kl_loss = (1 +z_log_var - torch.square(z_mean-pz_mean) - z_log_var.exp())
        kl_loss = -0.5*kl_loss.sum(-1).cuda()
        
        
        sigma = torch.exp(0.5 * r_log_var)
        regressor_loss=-0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (r_mean-inputs_r)**2
        regressor_loss=-regressor_loss.mean()
        
        return regressor_loss



