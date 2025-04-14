import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0, e, i):
        self.F = F
        self.H = H  
        self.Q = Q  
        self.R = R  
        self.x = x0  
        self.P = P0  
        self.e = e
        self.i = i
        self.c = 0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x
    
    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        self.e = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, self.e)
        self.i = self.i + self.e

        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

        return self.x, self.e
    
class KalmanController(KalmanFilter):
    def __init__(self, F, H, Q, R, x0, P0, e, i, ki, T):  
        super().__init__(F, H, Q, R, x0, P0, e, i)
        self.ki = ki
        self.T = T
        self.f = None
    
    def nforce(self, tgf):
        self.predict()
        self.update(tgf)
        self.f = self.ki * np.abs(self.T * (self.i[0][0]-self.e[0][0]) + (self.T/2)*self.e[0][0])/2
    
    def forces(self,tgf, dir):
        self.nforce(tgf)
        force= np.array([dir*self.f])
        torque= np.array([0, 0, 0])
        return force[0], torque[0], self.f, self.e