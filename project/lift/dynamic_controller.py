import numpy as np

class DynamicController:
    def __init__(self, ki, T):
        self.ki= ki
        self.T= T
        self.df0= None
        self.df1= None

    def compute_force(self, xh0, eh0, xh1, eh1 ):

        self.df0= self.ki* np.abs(self.T*(xh0-eh0) + (self.T/2)*eh0)/2
        self.df1= self.ki* np.abs(self.T*(xh1-eh1) + (self.T/2)*eh1)/2
        return self.df0
    

    def compute_action(self, y0, y1):

        force=  np.array([y0*self.df0, -y1*self.df1])
        torque= np.array([[0, 0, 0],[0, 0, 0]])
        return force, torque
