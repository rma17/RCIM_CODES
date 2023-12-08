# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:53:25 2021

@author: mrd

This is the main script for simulating the RCIM paper's algorithm in 4 out of 5 experiment

Class UR10 consists the main functions that can observe from simulation and control the robot:
    _init_: obtain the handles in simulations
    suction,drop: functions that control the suction cap in the robot
    reset1: the main function that simulate the initial scenarios including CASE 1 and CASE 2 as stated in the paper
    obj_importance: high-level decesion module for object selection in different stages
    dummy_obs: obtain observations from Vrep
    produce_subgoal: subgoals for low-level motion 
    action1: render the produced subgoal in simulation
    produce_action: joint actions for low-level motion 
    dummy_step: control the robot by produced joint actions


manipulation: the main function for the TAMP system

"""


import math
import time
import vrep
import numpy as np
from MLP_VAE import VAE_Linear
import torch
from dynamics import MLP
from same3 import GNN
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GCN=GNN(hidden_channels=128)
GCN.load_state_dict(torch.load('correction3.pt'))
GCN=GCN.to(device)
GCN.eval()

class UR10:
    # variates
    joint_angle = [0,0,0,0,0,0]     # each angle of joint
    RAD2DEG = 180 / math.pi         # transform radian to degrees
    
    # Handles information
    jointNum = 6
    baseName = 'UR10'
    rgName = 'Suction'
    jointName = 'UR10_joint'  
    subgoal='subgoal'
    cuboid_name1='Cuboid1'
    cuboid_name2='Cuboid2'
    cuboid_name3='Cuboid3'
    cuboid_name4='Cuboid4'
    cuboid_name5='Cuboid5'
    
    obs_name='Obs1'
    #target space
    cuboid1='Cube1'
    cuboid2='Cube2'
    cuboid3='Cube3'
    cuboid4='Cube4'
    cuboid5='Cube5'
    
    
    dummy_2='dummy_2'
    dummy_3='Obs3'
    link_name='UR10_link7'
    gripper_tip='tip'
    force_snesor='UR10_connection'
    box_pos='box_pos'
    
    # communication and read the handles
    def __init__(self):
        jointNum = self.jointNum
        baseName = self.baseName
        rgName = self.rgName
        jointName = self.jointName
        gripper_tip=self.gripper_tip
        print('Simulation started')
        vrep.simxFinish(-1)    
      
        while True:
            # connect the vrep
            clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            if clientID > -1:
                print("Connection success!")
                break
            else:
                time.sleep(0.2)
                print("Failed connecting to remote API server!")
                print("Maybe you forget to run the simulation on vrep...")

        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot) 
        jointHandle = [0]*jointNum
        for i in range(jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i+1), vrep.simx_opmode_blocking)
            jointHandle[i] = returnHandle

        _, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)
        _, rgHandle = vrep.simxGetObjectHandle(clientID, rgName, vrep.simx_opmode_blocking)
      
        
        
        
        
        _,cuboid_handle1=vrep.simxGetObjectHandle(clientID, self.cuboid_name1, vrep.simx_opmode_blocking)
        _,cuboid_handle2=vrep.simxGetObjectHandle(clientID,self.cuboid_name2,vrep.simx_opmode_blocking)
        _,cuboid_handle3=vrep.simxGetObjectHandle(clientID,self.cuboid_name3,vrep.simx_opmode_blocking)
        _,self.cuboid_handle4=vrep.simxGetObjectHandle(clientID,self.cuboid_name4,vrep.simx_opmode_blocking)
        _,self.cuboid_handle5=vrep.simxGetObjectHandle(clientID,self.cuboid_name5,vrep.simx_opmode_blocking)
        
        
        _,self.obs1_handle=vrep.simxGetObjectHandle(clientID,self.obs_name,vrep.simx_opmode_blocking)
        
        _,self.cube_handle1=vrep.simxGetObjectHandle(clientID, self.cuboid1, vrep.simx_opmode_blocking)
        _,self.cube_handle2=vrep.simxGetObjectHandle(clientID, self.cuboid2, vrep.simx_opmode_blocking)
        _,self.cube_handle3=vrep.simxGetObjectHandle(clientID, self.cuboid3, vrep.simx_opmode_blocking)
        _,self.cube_handle4=vrep.simxGetObjectHandle(clientID, self.cuboid4, vrep.simx_opmode_blocking)
        _,self.cube_handle5=vrep.simxGetObjectHandle(clientID, self.cuboid5, vrep.simx_opmode_blocking)
        
        
        _,self.box=vrep.simxGetObjectHandle(clientID, self.box_pos, vrep.simx_opmode_blocking)
        
        _,link_handle=vrep.simxGetObjectHandle(clientID,self.link_name,vrep.simx_opmode_blocking)
        
    
        _,tipHandle=vrep.simxGetObjectHandle(clientID, gripper_tip, vrep.simx_opmode_blocking)
       
        _,render_handle=vrep.simxGetObjectHandle(clientID, self.subgoal, vrep.simx_opmode_blocking)
        
        
        _,self.dum_2=vrep.simxGetObjectHandle(clientID, self.dummy_2, vrep.simx_opmode_blocking)
        _,self.dum_3=vrep.simxGetObjectHandle(clientID, self.dummy_3, vrep.simx_opmode_blocking)
       
        jointConfig = np.zeros((jointNum, 1))
        
        for i in range(jointNum):
             _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_blocking)
             jointConfig[i] = jpos
          
        self.clientID = clientID
        self.jointHandle = jointHandle
        self.rgHandle = rgHandle
        self.jointConfig = jointConfig
        self.tipHandle=tipHandle
        self.lastCmdTime=vrep.simxGetLastCmdTime(self.clientID)
        self.baseHandle=baseHandle
        self.cuboid_handle1=cuboid_handle1
        self.cuboid_handle2=cuboid_handle2
        self.cuboid_handle3=cuboid_handle3
       
        self.link_handle=link_handle
        self.target_config=[]

        self.render_handle=render_handle
        self.track_pos=[0,0,0]
        self.track_orientation=[0,0,0]
        self.attention=-1
        self.goalpos=np.array([0,0,0,0,0,0])
        self.x=0
        self.reset_pos=[0,0,0]
        self.label=0
        self.pos=[]
        self.filled=[]
        self.se=[]
        
        
    def showJointAngles(self):
        jointNum = self.jointNum
        clientID = self.clientID
        jointHandle = self.jointHandle
        jpos1=[]
        for i in range(jointNum):
            _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_blocking)
            jpos1.append(jpos)
        return jpos1
   
    def rotateAllAngle(self, joint_angle):
        clientID = self.clientID
        jointNum = self.jointNum
        jointHandle = self.jointHandle
        threshold=0.01
        t=0
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(clientID, jointHandle[i], joint_angle[i], vrep.simx_opmode_oneshot)
        currenttheta=np.zeros((6,))
        while True:
            for i in range(6):
                _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_oneshot_wait)
                
                currenttheta[i]=jpos
            diff=currenttheta-np.array(joint_angle)
            #print(diff)
            tol=np.max(diff)
            t+=1
            #print(np.abs(tol))
            if np.abs(tol)<threshold:
               #print('step:',t)
               break
        
        self.jointConfig = joint_angle
    
    def rotateCertainAnglePositive(self, num, angle):#for a single angle rotation
        clientID = self.clientID
        jointHandle = self.jointHandle
        jointConfig = self.jointConfig
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], (jointConfig[num]+angle), vrep.simx_opmode_oneshot)
        jointConfig[num] = jointConfig[num] + angle
        
        self.jointConfig = jointConfig
    def rotateCertainAngleNegative(self, num, angle):#for a single angle rotation
        clientID = self.clientID
        
        jointHandle = self.jointHandle    
        temp=self.jointConfig
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], (temp[num]-angle), vrep.simx_opmode_oneshot)
        temp[num] = temp[num] - angle
        
       
    def rotateCertainAnglePositive1(self, num, angle):
        clientID = self.clientID
        
        jointHandle = self.jointHandle    
        temp=self.jointConfig
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], angle, vrep.simx_opmode_oneshot)
        # temp[num] = temp[num] - angle    
        
        
        
        self.jointConfig =temp
    def rest1(self,rand):  
       
        
        if rand==0:
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle1,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle2,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle3,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle4,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle5,-1,[0,0,0],vrep.simx_opmode_blocking)
            pos = list(np.random.uniform([-0.65,-0.45,0.2], [-0.7,-0.5,0.2]))
       
            goal_index = np.arange(4)
            np.random.shuffle(goal_index)
            goal=[pos,[pos[0]+0.1,pos[1],pos[2]],[pos[0]-0.12,pos[1],pos[2]],[pos[0],pos[1]-0.2,pos[2]]]
        
            demo_goal=[[-1.3,0.5,0.7348],[-1.3,0.6,0.7348],[-1.3,0.38,0.7348],[-1.1,0.5,0.738]]
            self.target_config=[goal[goal_index[0]],goal[goal_index[1]],goal[goal_index[2]],goal[goal_index[3]]]
       
            vrep.simxSetObjectPosition(self.clientID,self.box,self.baseHandle,pos,vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.clientID,self.box,-1,[0,0,0],vrep.simx_opmode_blocking)
            
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle1,self.baseHandle,[-0.16348,	-0.62,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle2,self.baseHandle,[-0.11,  -0.5087095,  0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle3,self.baseHandle,[-0.06,	-0.62,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle4,self.baseHandle,[-0.21,	-0.53,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle5,self.baseHandle,[-0.3,	-0.6,	0.2],vrep.simx_opmode_blocking)
            
            
            demo_handle=[self.cube_handle1,self.cube_handle2,self.cube_handle3,self.cube_handle4,self.cube_handle5]
            a=[0,1,2,3,4]
            self.se=random.sample(a, 4)
           
            self.se.sort()
            
           
            
            
            
            for i in range(4):
                vrep.simxSetObjectPosition(self.clientID,demo_handle[self.se[i]],-1,demo_goal[goal_index[i]],vrep.simx_opmode_blocking)
            
            # self.target_config=[pos[0],pos[1],pos[2]]
        if rand==1:
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle1,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle2,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle3,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle4,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle5,-1,[0,0,0],vrep.simx_opmode_blocking)
            pos = list(np.random.uniform([-0.65,-0.45,0.2], [-0.7,-0.5,0.2]))
       
            goal_index = np.arange(4)
            np.random.shuffle(goal_index)
            goal=[pos,[pos[0]+0.1,pos[1],pos[2]],[pos[0]-0.12,pos[1],pos[2]],[pos[0],pos[1]-0.2,pos[2]]]
        
            demo_goal=[[-1.3,0.5,0.7348],[-1.3,0.6,0.7348],[-1.3,0.38,0.7348],[-1.1,0.5,0.738]]
            self.target_config=[goal[goal_index[0]],goal[goal_index[1]],goal[goal_index[2]],goal[goal_index[3]]]
       
            vrep.simxSetObjectPosition(self.clientID,self.box,self.baseHandle,pos,vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.clientID,self.box,-1,[0,0,0],vrep.simx_opmode_blocking)
            
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle1,self.baseHandle,[-0.16348,	-0.62,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle2,self.baseHandle,[-0.11,  -0.5087095,  0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle3,self.baseHandle,[-0.04,	-0.52,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle4,self.baseHandle,[-0.21,	-0.53,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle5,self.baseHandle,[-0.3,	-0.6,	0.2],vrep.simx_opmode_blocking)
            
            
            
            
            demo_handle=[self.cube_handle1,self.cube_handle2,self.cube_handle3,self.cube_handle4,self.cube_handle5]
            a=[0,1,2,3,4]
            self.se=random.sample(a, 4)
            self.se.sort()
            self.se=[0,1,2,4]
            ach=random.sample(self.se,1)
            ach=[4]
           
            index=self.se.index(ach[0])
            
            
            
            obj_handle=[self.cuboid_handle1,self.cuboid_handle2,self.cuboid_handle3,self.cuboid_handle4,self.cuboid_handle5]
            vrep.simxSetObjectPosition(self.clientID,obj_handle[4],self.baseHandle,self.target_config[index],vrep.simx_opmode_blocking)
            
            for i in range(4):
                vrep.simxSetObjectPosition(self.clientID,demo_handle[self.se[i]],-1,demo_goal[goal_index[i]],vrep.simx_opmode_blocking)
        if rand==2:
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle1,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle2,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle3,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle4,-1,[0,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle5,-1,[0,0,0],vrep.simx_opmode_blocking)
            pos = list(np.random.uniform([-0.65,-0.45,0.2], [-0.7,-0.5,0.2]))
       
            goal_index = np.arange(4)
            np.random.shuffle(goal_index)
            goal=[pos,[pos[0]+0.1,pos[1],pos[2]],[pos[0]-0.12,pos[1],pos[2]],[pos[0],pos[1]-0.2,pos[2]]]
        
            demo_goal=[[-1.3,0.5,0.7348],[-1.3,0.6,0.7348],[-1.3,0.4,0.7348],[-1.1,0.5,0.738]]
            self.target_config=[goal[goal_index[0]],goal[goal_index[1]],goal[goal_index[2]],goal[goal_index[3]]]
       
            vrep.simxSetObjectPosition(self.clientID,self.box,self.baseHandle,pos,vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.clientID,self.box,-1,[0,0,0],vrep.simx_opmode_blocking)
            
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle1,self.baseHandle,[-0.16348,	-0.62,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle2,self.baseHandle,[-0.11,  -0.5087095,  0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle3,self.baseHandle,[-0.0463,	-0.6261,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle4,self.baseHandle,[-0.21,	-0.53,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle5,self.baseHandle,[-0.3,	-0.6,	0.2],vrep.simx_opmode_blocking)
            
            
            
            
            demo_handle=[self.cube_handle1,self.cube_handle2,self.cube_handle3,self.cube_handle4,self.cube_handle5]
            a=[0,1,2,3,4]
            self.se=random.sample(a, 4)
            self.se=[0,1,2,4]
            self.se.sort()
            ach=random.sample(self.se,2)
            ach=[1,4]
            ach.sort()
            index=[self.se.index(ach[i]) for i in range(2)]
            
            
            
            obj_handle=[self.cuboid_handle1,self.cuboid_handle2,self.cuboid_handle3,self.cuboid_handle4,self.cuboid_handle5]
            for i in range(2):
              vrep.simxSetObjectPosition(self.clientID,obj_handle[ach[i]],self.baseHandle,self.target_config[index[i]],vrep.simx_opmode_blocking)
            
            for i in range(4):
                vrep.simxSetObjectPosition(self.clientID,demo_handle[self.se[i]],-1,demo_goal[goal_index[i]],vrep.simx_opmode_blocking)
        if rand==3:
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle1,-1,[2.5,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle2,-1,[2.5,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle3,-1,[2.5,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle4,-1,[2.5,0,0],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cube_handle5,-1,[2.5,0,0],vrep.simx_opmode_blocking)
            pos = list(np.random.uniform([-0.65,-0.45,0.2], [-0.7,-0.5,0.2]))
       
            goal_index = np.arange(4)
            np.random.shuffle(goal_index)
            goal=[pos,[pos[0]+0.1,pos[1],pos[2]],[pos[0]-0.12,pos[1],pos[2]],[pos[0],pos[1]-0.2,pos[2]]]
            goal_index=[3,1,0,2]
            demo_goal=[[-1.3,0.5,0.7348],[-1.3,0.6,0.7348],[-1.3,0.4,0.7348],[-1.1,0.5,0.738]]
            self.target_config=[goal[goal_index[0]],goal[goal_index[1]],goal[goal_index[2]],goal[goal_index[3]]]
       
            vrep.simxSetObjectPosition(self.clientID,self.box,self.baseHandle,pos,vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.clientID,self.box,-1,[0,0,0],vrep.simx_opmode_blocking)
            
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle1,self.baseHandle,[-0.16348,	-0.62,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle2,self.baseHandle,[-0.11,  -0.5087095,  0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle3,self.baseHandle,[-0.0463,	-0.6261,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle4,self.baseHandle,[-0.21,	-0.53,	0.2],vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.clientID,self.cuboid_handle5,self.baseHandle,[-0.3,	-0.6,	0.2],vrep.simx_opmode_blocking)
            
            
            
            demo_handle=[self.cube_handle1,self.cube_handle2,self.cube_handle3,self.cube_handle4,self.cube_handle5]
            a=[0,1,2,3,4]
            self.se=random.sample(a, 4)
            self.se.sort()
            ach=random.sample(self.se,3)
            ach.sort()
            
            
            self.se=[0,1,2,4]
            ach=[1,2,4]
            index=[self.se.index(ach[i]) for i in range(3)]
            
            
            
            obj_handle=[self.cuboid_handle1,self.cuboid_handle2,self.cuboid_handle3,self.cuboid_handle4,self.cuboid_handle5]
            for i in range(3):
              vrep.simxSetObjectPosition(self.clientID,obj_handle[ach[i]],self.baseHandle,self.target_config[index[i]],vrep.simx_opmode_blocking)
            
            for i in range(4):
                vrep.simxSetObjectPosition(self.clientID,demo_handle[self.se[i]],-1,demo_goal[goal_index[i]],vrep.simx_opmode_blocking)
        
          
            
        ###############################################################################################
        
        vrep.simxSetObjectOrientation(self.clientID,self.cuboid_handle1,-1,[0,	0,	0],vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID,self.cuboid_handle2,-1,[0,  0,  0],vrep.simx_opmode_blocking)  
        vrep.simxSetObjectOrientation(self.clientID,self.cuboid_handle3,-1,[0,  0,  0],vrep.simx_opmode_blocking) 
        vrep.simxSetObjectOrientation(self.clientID,self.cuboid_handle4,-1,[0,	0,	0],vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID,self.cuboid_handle5,-1,[0,  0,  0],vrep.simx_opmode_blocking)  
        # vrep.simxSetObjectOrientation(self.clientID,self.cuboid_handle3,-1,[0,  0,  0],vrep.simx_opmode_blocking) 
        
        
        
        
        
        vrep.simxSetObjectOrientation(self.clientID,self.cube_handle1,-1,[0,	0,	0],vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.clientID,self.cube_handle2,-1,[0,  0,  0],vrep.simx_opmode_blocking)  
        vrep.simxSetObjectOrientation(self.clientID,self.cube_handle3,-1,[0,  0,  0],vrep.simx_opmode_blocking) 
        vrep.simxSetObjectOrientation(self.clientID,self.cube_handle4,-1,[0,  0,  0],vrep.simx_opmode_blocking)    
        vrep.simxSetObjectOrientation(self.clientID,self.cube_handle5,-1,[0,  0,  0],vrep.simx_opmode_blocking)   
           
           
       
        
        
        _,pos=vrep.simxGetObjectPosition(self.clientID,self.rgHandle,-1,vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.clientID,self.render_handle,-1,pos,vrep.simx_opmode_blocking)
    
        
    def action1(self,action):
        vrep.simxSetObjectPosition(self.clientID, self.render_handle, self.baseHandle, action, vrep.simx_opmode_blocking)
    def suction(self):
        emptyBuff = bytearray()
       
        res,retInts,resultPos,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'test',vrep.sim_scripttype_childscript,'activateSuctionPad',[],[],[],emptyBuff,vrep.simx_opmode_blocking)
  
    def drop(self):
        emptyBuff = bytearray()
         
        res,retInts,resultPos,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'test',vrep.sim_scripttype_childscript,'deactivateSuctionPad',[],[],[],emptyBuff,vrep.simx_opmode_blocking)
    def goal_distance(self,goal_a, goal_b):
    
      assert goal_a.shape == goal_b.shape
      dis=np.linalg.norm(goal_a - goal_b, axis=-1)
      return (dis > 0.06)
   
    def obj_importance(self,select):
       
       _,pos1=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle1,self.baseHandle,vrep.simx_opmode_blocking)
       
       
       _,pos2=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle2,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos3=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle3,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos6=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle4,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos7=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle5,self.baseHandle,vrep.simx_opmode_blocking)
       
       obj_obs=[pos1,pos2,pos3,pos6,pos7]
       
       
       obs=[obj_obs[self.se[i]] for i in range(4)]
       
       
       
       
       full_filled=[0,0,0,0,0]
       filled=[0 if self.goal_distance(np.array(obs[i]), np.array(self.target_config[i])) else 1 for i in range(4)]
       for i in range(4):
             full_filled[self.se[i]]=filled[i]
       self.filled=full_filled
       
       selected=[0,0,0,0,0]
       for i in range(4):
                 selected[self.se[i]]=1
       x=[[pos1[0],pos1[1],pos1[2],selected[0]],
          [pos2[0],pos2[1],pos2[2],selected[1]],
          [pos3[0],pos3[1],pos3[2],selected[2]],
          [pos6[0],pos6[1],pos6[2],selected[3]],      
          [pos7[0],pos7[1],pos7[2],selected[4]]] 
       x=torch.tensor(x).to(device)
       batch=[0]*x.size()[0]
       index=[[0,1,2,3],[1,2,3,4]]
       index=torch.tensor(index).to(device)
       batch=torch.tensor(batch).to(device)
       
       goal=[[0]*3 for _ in range(5)]
       
       
       for i in range(4):
           goal[self.se[i]]=self.target_config[i]
       goal=torch.tensor(goal).to(device)
       if select==0:
           GCN.select(x,index,batch)
       
       importance,label=GCN(x,index,batch,goal)
       
       self.importance=importance.argmax(dim=1)
       
       label=label.argmax(dim=1)
       self.label=torch.nn.functional.one_hot(label,num_classes=5)
       
    def dummy_obs(self,select):
       _,pos0=vrep.simxGetObjectPosition(self.clientID,self.box,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos1=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle1,self.baseHandle,vrep.simx_opmode_blocking)
       
       
       _,pos2=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle2,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos3=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle3,self.baseHandle,vrep.simx_opmode_blocking)
       
       
       _,pos4=vrep.simxGetObjectPosition(self.clientID,self.render_handle,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos5=vrep.simxGetObjectPosition(self.clientID,self.rgHandle,self.baseHandle,vrep.simx_opmode_blocking)
       
       
       _,pos6=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle4,self.baseHandle,vrep.simx_opmode_blocking)
       _,pos7=vrep.simxGetObjectPosition(self.clientID,self.cuboid_handle5,self.baseHandle,vrep.simx_opmode_blocking)
       
       obj_obs=[pos1,pos2,pos3,pos6,pos7]
       
       
       obs=[obj_obs[self.se[i]] for i in range(3)]
 
       full_obs=[pos0,pos1,pos2,pos3,pos6,pos7]

       obs=np.array([full_obs[self.importance]+pos4])

       return obs,pos5,self.label
   
    def dummy_step(self,pos,label):
        pos=pos*math.pi/180
        
        self.rotateCertainAnglePositive(0,pos[0])
        self.rotateCertainAnglePositive(1,pos[1])
        self.rotateCertainAnglePositive(2,pos[2])
        self.rotateCertainAnglePositive1(5,pos[3])
        return self.dummy_obs(label)
    def produce_subgoal(self,obs,label):
          obs=torch.tensor(obs).to(device)
          out=vae(obs.float(),label.float())[5]
          out=out.cpu().detach().numpy()
          out=out.reshape(-1)
          out[2]=max(out[2],0.258)
          return out
    def produce_actions(self,diff):
              diff=torch.tensor(diff).to(device)
              action=dynamics(diff.float())
              action= action.cpu().detach().numpy()
              action=action.reshape(-1)
              return action
          
def rest(agent):
        reset_joint=[1.26705908775330,-0.140426874160767,-1.77047705650330,0.339693784713745,1.57081580162048,1.26711893081665]
        agent.drop()
        time.sleep(0.5)
        
        agent.rotateAllAngle(reset_joint)
        agent.rest1(6)

robot = UR10()
vae=VAE_Linear(6,3,6,5) 
vae.load_state_dict(torch.load('sub1.pt'))  
vae.to(device)
vae.eval()

dynamics=MLP(3, 4, 3,128)
dynamics.load_state_dict(torch.load('direction_1.pt'))
dynamics.eval()
dynamics.to(device)




def manipulation(): 
 
        #Picking loop
        robot.suction()
        robot.obj_importance(0)
        obs,tip,label=robot.dummy_obs(1)
        
        for _ in range(3):
          
          obs,tip,_=robot.dummy_obs(3)
          out=robot.produce_subgoal(obs, label)
         
          
          robot.action1(out)
         
          for _ in range(20):
              
              diff=out-tip
              if(np.linalg.norm(diff)<0.03):
                 
                  break
              action=robot.produce_actions(diff)
              
              _,tip,_=robot.dummy_step(action,1)
              
              
        #Palcing loop
        robot.obj_importance(1)
        for _ in range(3):
          
          obs,tip,label=robot.dummy_obs(0)
          
          out=robot.produce_subgoal(obs, label)

          robot.action1(out)
          
          for _ in range(20):
              diff=out-tip
              if(np.linalg.norm(diff)<0.03):
                 
                  break
              action=robot.produce_actions(diff)
              
              _,tip,_=robot.dummy_step(action,1)

        robot.drop()

c=0

for count in range(1):
    rest(robot)
    ran=np.random.randint(4)
    robot.rest1(ran)
   
    for _ in range(4):
        
        rest(robot)
       
        time_start=time.time()
        manipulation()
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        robot.obj_importance(1)
        a=[robot.filled[robot.se[i]] for i in range(4)]
        if all(a)==1:
            
          c+=1
          print(c,"/",count)
          break
   
   


