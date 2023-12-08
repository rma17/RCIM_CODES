# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:56:11 2021

@author: mrd
This the function for nueral dynamic planner
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import pandas as pd
from sklearn.model_selection import KFold
class MLP(nn.Module):
    '''A simple implementation of the multi-layer neural network'''
    def __init__(self, n_input=3, n_output=3, n_h=2, size_h=64):
        '''
        Specify the neural network architecture

        :param n_input: The dimension of the input
        :param n_output: The dimension of the output
        :param n_h: The number of the hidden layer
        :param size_h: The dimension of the hidden layer
        '''
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        assert n_h >= 1, "h must be integer and >= 1"
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.tanh(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.tanh(out)
        out = self.fc_out(out)
        return out

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)
class DynamicModel(object):
    '''Neural network dynamic model '''
    def __init__(self):
       
        
        
        
        self.model = MLP(6, 4, 3,128)
        
        
        
        self.n_epochs = 1000
        self.lr = 0.001
        self.batch_size = 32
        
        
       
        self.criterion =nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, X,Y):
       
        # Normalize the dataset and record data distribution (mean and std)
     #   datasets, labels = self.norm_train_data(trainset["data"],trainset["label"])
        X=torch.tensor(X).float()
        Y=torch.tensor(Y).float()
        Dataset1=TensorDataset(X,Y)
        trainloader=DataLoader(dataset=Dataset1,batch_size=self.batch_size,shuffle=True)
        total_step = len(trainloader)
        print(f"Total training step per epoch [{total_step}]")
        loss_epochs = []
        for epoch in range(1, 500):
            loss_this_epoch = []
            for i, (datas, labels) in enumerate(trainloader):
                # datas = self.Variable(torch.FloatTensor(np.float32(datas)))
                # labels = self.Variable(torch.FloatTensor(np.float32(labels)))
               
              #  print(labels.squeeze().shape)
                self.optimizer.zero_grad()
                outputs = self.model(datas)
               # labels=labels.squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())
            loss_epochs.append(np.mean(loss_this_epoch))
       
               
               
              
            
        return loss_epochs
    def validate_model(self, X, Y):
        '''
        Validate the trained model

        :param datasets: (numpy array) input data
        :param labels: (numpy array) corresponding label
        :return: average loss
        '''
        X=torch.tensor(X).float()
        Y=torch.tensor(Y).float()
        Dataset1=TensorDataset(X,Y)
        test_loader=DataLoader(dataset=Dataset1,batch_size=self.batch_size,shuffle=True)
        loss_list = []
        for i, (datas, labels) in enumerate(test_loader):
            # datas = self.Variable(torch.FloatTensor(np.float32(datas)))
            # labels = self.Variable(torch.FloatTensor(np.float32(labels)))
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
        loss_avr = np.average(loss_list)
        print('loss:',loss_avr)  
        return loss_avr
    def predict(self,x):
         x = np.array(x)
      #  x = self.pre_process(x)
         with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0) # not sure here
         out_tensor = self.model(x_tensor)
         out = out_tensor.cpu().detach().numpy()
       # out = self.after_process(out)
         return out
    def save(self):
         torch.save(self.model.state_dict(), 'direction_2.pt')
