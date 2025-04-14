import numpy as np
from robosuite.utils.control_utils import *
from robosuite.utils.transform_utils import *

class PoseController:
    def __init__(self, robots, kp=0.9):
        self.robots= robots
        self.kp= kp
        self.joint_torques= None

    def set(self, robots):
        self.robots= robots
        self.joint_torques= np.array([0])

    def compute(self, robots, desired_pos, desired_ori):
        self.set(robots)
        for i in range (len(self.robots)):
            KP = robots[i].controller.nums2array(self.kp, 6)
            KD = 2 * np.sqrt(KP) * 1

            pos= robots[i].controller.ee_pos
            ori= robots[i].controller.ee_ori_mat
            
            pos_e= desired_pos[i] - pos
            ori_e= orientation_error(desired_ori[i], ori)
            velpos_e = - robots[i].controller.ee_pos_vel
            velori_e = - robots[i].controller.ee_ori_vel
    
            force_d = np.multiply(np.array(pos_e), np.array(KP[0:3])) + np.multiply(velpos_e, KD[0:3])
            torque_d= np.multiply(np.array(ori_e), np.array(KP[3:6])) + np.multiply(velori_e, KD[3:6])

            _, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
                robots[i].controller.mass_matrix,
                robots[i].controller.J_full,
                robots[i].controller.J_pos,
                robots[i].controller.J_ori
            )

            decoupled_force = np.dot(lambda_pos, force_d)
            decoupled_torque = np.dot(lambda_ori, torque_d)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])

            torques = np.dot(robots[i].controller.J_full.T, decoupled_wrench) 

            torques += nullspace_torques(
                            robots[i].controller.mass_matrix,
                            nullspace_matrix, 
                            robots[i].controller.initial_joint, 
                            robots[i].controller.joint_pos, 
                            robots[i].controller.joint_vel
            )   
            self.joint_torques= np.append(self.joint_torques,torques)

        return np.delete(self.joint_torques,0)