import numpy as np
from robosuite.project.reorientation.force_control import *
from robosuite.project.reorientation.kalman_filter import *
from robosuite.project.reorientation.coulomb_controller import *
from robosuite.project.reorientation.transformations import *



class ReactiveLayer():
    def __init__(self, mu_, kp_, F_, H_, Q_, R_, x0_, P0_, ki_, T_):
        self.coulombCtrl0 = CoulombController(mu_, kp_)
        self.coulombCtrl1 = CoulombController(mu_, kp_)
        self.kalmanCtrl0 = KalmanController(F_, H_, Q_, R_, x0_, P0_, e=0, i=0, ki=ki_, T= T_)
        self.kalmanCtrl1 = KalmanController(F_, H_, Q_, R_, x0_, P0_, e=0, i=0, ki=ki_, T= T_)
        self.tgf0 = 0
        self.tgf1 = 0
        self.e0 = 0
        self.e1 = 0
        self.n0 = None
        self.n1 = None
        self.f0 = None
        self.f1 = None
        self.act = None
        self.ReL_intervention = None
        self.df = None

    # function to set the measured tangential forces
    def measure_tangential_force(self, tgf0, tgf1):
        self.tgf0 = tgf0
        self.tgf1 = tgf1
    
    # function to set the actuation force direction
    def estimate_actuation_direction(self, n0, n1):
        self.n0 = n0
        self.n1 = n1

    # function to update KF residuals
    def update_residual(self):
        self.kalmanCtrl0.predict()
        self.kalmanCtrl1.predict()
        self.kalmanCtrl0.update(self.tgf0)
        self.kalmanCtrl1.update(self.tgf1)

    # function to compute the
    # robot's fingers forces
    def compute_force(self):
        _,_,sf0_intervent = self.coulombCtrl0.forces(self.tgf0, self.n0)
        _,_,sf1_intervent = self.coulombCtrl1.forces(self.tgf1, self.n1)
        _,_,df0_intervent, self.e0 = self.kalmanCtrl0.forces(self.tgf0, self.n0)
        _,_,df1_intervent, self.e1 = self.kalmanCtrl1.forces(self.tgf1, self.n1)

        if sf0_intervent>=sf1_intervent:
            sf0 = np.array([self.n0*sf0_intervent])[0]
            sf1 = np.array([self.n1*sf0_intervent])[0]
        else:
            sf0 = np.array([self.n0*sf1_intervent])[0]
            sf1 = np.array([self.n1*sf1_intervent])[0]
 
        if df0_intervent>=df1_intervent:
            df0 = np.array([self.n0*(df0_intervent)])[0]
            df1 = np.array([self.n1*(df0_intervent)])[0]
            self.df = df0_intervent
        else:
            df0 = np.array([self.n0*(df1_intervent)])[0]
            df1 = np.array([self.n1*(df1_intervent)])[0] 
            self.df = df1_intervent         
        

        self.f0 = sf0 + df0 
        self.f1 = sf1 + df1 

        #self.f0 = sf0 
        #self.f1 = sf1 

        self.ReL_intervention = sf0_intervent + sf1_intervent + df0_intervent + df1_intervent
    

    def exert_additional_force(self, robots, mat0, mat1, t0, t1):
        f0 = np.array(t0*0.011)
        f1 = np.array(t1*0.011)
        forces_d=  np.array([f0, f1])
        torques_d= np.array([[0,0,0], [0,0,0]])
        act = desired_force_to_torque_EEF(robots, mat0, mat1, forces_d, torques_d)
        return act

    # function to compute robot joints torques
    # associated to the robot exerted forces
    def exert_force(self, robots, mat0, mat1, t0, t1):
        forces_d=  np.array([self.f0, self.f1])
        torques_d= np.array([[0,0,0], [0,0,0]])
        self.act = desired_force_to_torque_EEF(robots, mat0, mat1, forces_d, torques_d) + self.exert_additional_force(robots, mat0, mat1, t0, t1)

    # function to compute the action to exert in environment
    def action(self, robots, mat0, mat1, tgf0, tgf1, n0, n1, t0, t1):
        self.measure_tangential_force(tgf0, tgf1)
        self.estimate_actuation_direction(n0, n1)
        self.compute_force()
        self.exert_force(robots, mat0, mat1, t0, t1)
        return self.act, self.f0, self.f1, self.ReL_intervention, self.e0, self.e1, self.df
