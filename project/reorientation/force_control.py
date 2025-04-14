import numpy as np
from robosuite.utils.control_utils import *
from robosuite.utils.transform_utils import *


# function to generate joint torques given the desired forces and torques in the world reference frame
def desired_force_to_torque_W(env_robots,R_matrix0, R_matrix1, desired_forces, desired_torques):
    total_torques= np.array([0])

    # or alternatively the forces are already expressed in world reference frame
    desired_forces_0 = desired_forces[0,:]
    desired_forces_1 = desired_forces[1,:]
    desired_torques_0= desired_torques[0,:]
    desired_torques_1= desired_torques[1,:]  

    # compose single desired force vector and desired torque vector
    desired_forces=  np.array([desired_forces_0, desired_forces_1])
    desired_torques= np.array([desired_torques_0,desired_torques_1])

    # for each robots
    for i in range(2):

        # calculate relevant control algorithm matrices 
        _, lambda_pos, lambda_ori, _ = opspace_matrices(
            env_robots[i].controller.mass_matrix,
            env_robots[i].controller.J_full,
            env_robots[i].controller.J_pos,
            env_robots[i].controller.J_ori
        )

        # decouple desired positional control from ori control
        decoupled_force = np.dot(lambda_pos, desired_forces[i, :])
        decoupled_torque = np.dot(lambda_ori, desired_torques[i, :])
        decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])

        # compute final desired torques without gravity compensation  J^T * F
        torques = np.dot(env_robots[i].controller.J_full.T, decoupled_wrench)

        # append joint torques for i-th robot 
        total_torques= np.append(total_torques, torques)

    # delete first element of torques vector 
    total_torques= np.delete(total_torques, 0)

    return total_torques


# function to generate joint torques given the desired forces and torques in the eefs reference frame
def desired_force_to_torque_EEF(env_robots, R_matrix0, R_matrix1, desired_forces, desired_torques):
    total_torques= np.array([0])
    
    # compute forces/torques from eef0/eef1 in the world reference frame
    desired_forces_0 = np.matmul(R_matrix0, desired_forces[0,:])
    desired_forces_1 = np.matmul(R_matrix1, desired_forces[1,:])
    desired_torques_0= np.matmul(R_matrix0, desired_torques[0,:])
    desired_torques_1= np.matmul(R_matrix1, desired_torques[1,:])

    # compose single desired force vector and desired torque vector
    desired_forces=  np.array([desired_forces_0, desired_forces_1])
    desired_torques= np.array([desired_torques_0,desired_torques_1])

    # for each robots
    for i in range(2):

        # calculate relevant control algorithm matrices 
        _, lambda_pos, lambda_ori, _ = opspace_matrices(
            env_robots[i].controller.mass_matrix,
            env_robots[i].controller.J_full,
            env_robots[i].controller.J_pos,
            env_robots[i].controller.J_ori
        )

        # decouple desired positional control from ori control
        decoupled_force = np.dot(lambda_pos, desired_forces[i, :])
        decoupled_torque = np.dot(lambda_ori, desired_torques[i, :])
        decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])

        # compute final desired torques without gravity compensation  J^T * F
        torques = np.dot(env_robots[i].controller.J_full.T, decoupled_wrench)

        # append joint torques for i-th robot 
        total_torques= np.append(total_torques, torques)

    # delete first element of torques vector 
    total_torques= np.delete(total_torques, 0)

    return total_torques



def wrench2torques(env, wrench_d):

    joint_torques= np.array([0])
    mat0= env._eef0_xmat
    mat1= env._eef1_xmat

    force_d0= np.matmul(mat0, wrench_d[:3])
    force_d1= np.matmul(mat1, wrench_d[3:6])
    torque_d0= np.matmul(mat0, wrench_d[6:9])
    torque_d1= np.matmul(mat1, wrench_d[9:12])

    force_d= np.array([force_d0, force_d1])
    torque_d= np.array([torque_d0, torque_d1])

    for i in range (len(env.robots)):
        _, lambda_pos, lambda_ori, _ = opspace_matrices(
        env.robots[i].controller.mass_matrix,
        env.robots[i].controller.J_full,
        env.robots[i].controller.J_pos,
        env.robots[i].controller.J_ori
        )
        force_dec = np.dot(lambda_pos, force_d[i,:])
        torque_dec = np.dot(lambda_ori, torque_d[i,:])
        wrench_dec = np.concatenate([force_dec, torque_dec])
        torques = np.dot(env.robots[i].controller.J_full.T, wrench_dec)
        joint_torques= np.append(joint_torques,torques) 

    return np.delete(joint_torques,0)

def force2torques(env, forces_d):

    joint_torques= np.array([0])
    mat0= env._eef0_xmat
    mat1= env._eef1_xmat

    
    # compute forces/torques from eef0/eef1 in the world reference frame
    force_d0= np.matmul(mat0, forces_d[:3])
    force_d1= np.matmul(mat1, forces_d[3:6])
    torque_d0= np.array([0,0,0])
    torque_d1= np.array([0,0,0])

    force_d= np.array([force_d0, force_d1])
    torque_d= np.array([torque_d0, torque_d1])

    for i in range (len(env.robots)):
        _, lambda_pos, lambda_ori, _ = opspace_matrices(
        env.robots[i].controller.mass_matrix,
        env.robots[i].controller.J_full,
        env.robots[i].controller.J_pos,
        env.robots[i].controller.J_ori
        )
        force_dec = np.dot(lambda_pos, force_d[i,:])
        torque_dec = np.dot(lambda_ori, torque_d[i,:])
        wrench_dec = np.concatenate([force_dec, torque_dec])
        torques = np.dot(env.robots[i].controller.J_full.T, wrench_dec)
        joint_torques= np.append(joint_torques,torques)

    return np.delete(joint_torques,0)

def Wforce2torques(env, forces_d):

    joint_torques= np.array([0])
    force_d0= forces_d[:3]
    force_d1= forces_d[3:6]
    torque_d0= np.array([0,0,0])
    torque_d1= np.array([0,0,0])
    force_d= np.array([force_d0, force_d1])
    torque_d= np.array([torque_d0, torque_d1])

    for i in range (len(env.robots)):
        _, lambda_pos, lambda_ori, _ = opspace_matrices(
        env.robots[i].controller.mass_matrix,
        env.robots[i].controller.J_full,
        env.robots[i].controller.J_pos,
        env.robots[i].controller.J_ori
        )
        force_dec = np.dot(lambda_pos, force_d[i,:])
        torque_dec = np.dot(lambda_ori, torque_d[i,:])
        wrench_dec = np.concatenate([force_dec, torque_dec])
        torques = np.dot(env.robots[i].controller.J_full.T, wrench_dec)
        joint_torques= np.append(joint_torques,torques) 

    return np.delete(joint_torques,0)


