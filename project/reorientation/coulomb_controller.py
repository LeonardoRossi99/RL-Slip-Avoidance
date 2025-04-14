import numpy as np

class CoulombController():
    def __init__(self, friction, k):
        self.friction= friction
        self.k= k
        self.nf= None
    
    def nforce(self, tgf):
        self.nf= self.k*(tgf/self.friction)/2
    
    def forces(self, tgf, dir):
        self.nforce(tgf)
        force= np.array([dir*self.nf])
        torque= np.array([0, 0, 0])
        return force[0], torque[0], self.nf
    