import numpy as np
from robosuite.utils.transform_utils import *

class MeasurementsLayer():
    def __init__(self):
        self.t_to_measure= None
        self.mat0 = None
        self.mat1 = None
        self.matC = None
        self.fee0 = None
        self.fee1 = None
        self.mat0C= None
        self.mat1C= None
        self.init_tg0= None
        self.init_tg1= None
    
    # function to update 
    # measuraments variables
    def update(self, obs):
        #self.matC = quat2mat(obs["can_quat"])
        self.matC = quat2mat(obs["milk_quat"])
        self.mat0 = quat2mat(obs["robot0_eef_quat"])
        self.mat1 = quat2mat(obs["robot1_eef_quat"])
        self.fee0 = obs["gripper0_contact_force"]
        self.fee1 = obs["gripper1_contact_force"]

    # function to set the initial time
    # to start all the measuraments
    def set_time_to_measure(self, t):
        self.t_to_measure = t

    # function to compute the relative 
    # orientation between eefs and object
    def compute_relative_orientation(self):
        self.mat0C = np.dot(self.matC, self.mat0.T)
        self.mat1C = np.dot(self.matC, self.mat1.T)

    # function to compute the normal
    # vector to the object surface for
    # each of the  robot end effector
    def get_actuation_dir(self, obs):
        self.update(obs)
        self.compute_relative_orientation()
        y = np.array([0,1,0])
        v1 = np.dot(self.mat0C, y)
        v2 = np.dot(self.mat1C, y)
        dir1 =  v1/np.linalg.norm(v1)
        dir2 = -v2/np.linalg.norm(v2)
        return dir1, dir2 
    
    # function to compoute the parallel
    # vector to the object surface for 
    # each of the robot end effecto
    def get_tangential_force_dir(self, obs):
        self.update(obs)
        self.compute_relative_orientation()
        z = np.array([0,0,1])
        v1 = np.dot(self.mat0C, z)
        v2 = np.dot(self.mat1C, z)
        dir1 = v1/np.linalg.norm(v1)
        dir2 = v2/np.linalg.norm(v2)
        return dir1, dir2 

    
    # function to compute the tangential force 
    def get_tangent_forces(self, obs):
        self.update(obs)
        self.compute_relative_orientation()
        tg0= np.abs(np.dot(self.mat0C.T, self.fee0)[2])
        tg1= np.abs(np.dot(self.mat1C.T, self.fee1)[2])
        return tg0, tg1
        
    # function to measure the  tangential force
    # befor the grasping of object 
    def set_init_tangent_forces(self, t, obs):
        if t==self.t_to_measure:
            self.init_tg0, self.init_tg1 = self.get_tangent_forces(obs)
    
    # function to compute the tangential force
    def measure_tangent_forces(self, t, obs):
        self.update(obs)
        if t< self.t_to_measure:
            tg0 = 0
            tg1 = 0
        else:
            self.set_init_tangent_forces(t,obs)
            tg0, tg1 = self.get_tangent_forces(obs)
            tg0 = tg0 - self.init_tg0
            tg1 = tg1 - self.init_tg1
        return tg0, tg1