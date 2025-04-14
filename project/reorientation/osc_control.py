import numpy as np
from robosuite.utils.control_utils import *
from robosuite.utils.transform_utils import *

def desired_position_to_torque(env, desired_position):

    env_robots = env.robots

    # to store the total torques
    total_torques= np.array([0])
    desired_pos = [desired_position[:3], desired_position[3:]] 

    # for each robots
    for i in range(2):

        # compute proportional and derivative controller gain
        kp = env_robots[i].controller.nums2array(0.90, 6)
        kd = 2 * np.sqrt(kp) * 1

        # store eef current position and orientation 
        current_pos= env_robots[i].controller.ee_pos
        current_ori= env_robots[i].controller.ee_ori_mat

        # compute position orientation error
        error_pos= desired_pos[i] - current_pos
        error_ori= orientation_error(current_ori, current_ori)

        # compute velocity errors assuming zero velocity goal
        error_vel_pos = - env_robots[i].controller.ee_pos_vel
        error_vel_ori = - env_robots[i].controller.ee_ori_vel

        # compute desired force based on position error and velocity error: F = kp* error_pos + kd* error_vel_pos
        desired_force = np.multiply(np.array(error_pos), np.array(kp[0:3])) + np.multiply(error_vel_pos, kd[0:3])

        # compute desired torque based on orientation error and velocity: tau= kp* error_ori + kd* error_vel_ori
        desired_torque = np.multiply(np.array(error_ori), np.array(kp[3:6])) + np.multiply(error_vel_ori, kd[3:6])

        # calculate relevant control algorithm matrices 
        _, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            env_robots[i].controller.mass_matrix,
            env_robots[i].controller.J_full,
            env_robots[i].controller.J_pos,
            env_robots[i].controller.J_ori
        )

        # decouple desired positional control from ori control
        decoupled_force = np.dot(lambda_pos, desired_force)
        decoupled_torque = np.dot(lambda_ori, desired_torque)
        decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])

        # compute final desired torques without gravity compensation: J^T * F
        torques = np.dot(env_robots[i].controller.J_full.T, decoupled_wrench) 

        # then add nullspace torques
        torques += nullspace_torques(
                                    env_robots[i].controller.mass_matrix,
                                    nullspace_matrix, 
                                    env_robots[i].controller.initial_joint, 
                                    env_robots[i].controller.joint_pos, 
                                    env_robots[i].controller.joint_vel
        )

        # append joint torques for i-th robot 
        total_torques= np.append(total_torques, torques)

    # delete first element of torques vector 
    total_torques= np.delete(total_torques, 0)

    return total_torques



def desired_pose_to_torque(env, desired_pos_ori):

    env_robots = env.robots

    # to store the total torques
    total_torques= np.array([0])
    
    desired_pos_v = desired_pos_ori[:6]
    desired_ori_v = desired_pos_ori[6:]

    desired_pos = [desired_pos_v[:3], desired_pos_v[3:]]
    desired_ori = [desired_ori_v[:4], desired_ori_v[4:]] 

    # for each robots
    for i in range(2):

        # compute proportional and derivative controller gain
        kp = env_robots[i].controller.nums2array(0.90, 6)
        kd = 2 * np.sqrt(kp) * 1

        # store eef current position and orientation 
        current_pos= env_robots[i].controller.ee_pos
        current_ori= env_robots[i].controller.ee_ori_mat

        # compute position orientation error
        error_pos= desired_pos[i] - current_pos
        error_ori= orientation_error(quat2mat(desired_ori[i]), current_ori)

        # compute velocity errors assuming zero velocity goal
        error_vel_pos = - env_robots[i].controller.ee_pos_vel
        error_vel_ori = - env_robots[i].controller.ee_ori_vel

        # compute desired force based on position error and velocity error: F = kp* error_pos + kd* error_vel_pos
        desired_force = np.multiply(np.array(error_pos), np.array(kp[0:3])) + np.multiply(error_vel_pos, kd[0:3])

        # compute desired torque based on orientation error and velocity: tau= kp* error_ori + kd* error_vel_ori
        desired_torque = np.multiply(np.array(error_ori), np.array(kp[3:6])) + np.multiply(error_vel_ori, kd[3:6])

        # calculate relevant control algorithm matrices 
        _, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            env_robots[i].controller.mass_matrix,
            env_robots[i].controller.J_full,
            env_robots[i].controller.J_pos,
            env_robots[i].controller.J_ori
        )

        # decouple desired positional control from ori control
        decoupled_force = np.dot(lambda_pos, desired_force)
        decoupled_torque = np.dot(lambda_ori, desired_torque)
        decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])

        # compute final desired torques without gravity compensation: J^T * F
        torques = np.dot(env_robots[i].controller.J_full.T, decoupled_wrench) 

        # then add nullspace torques
        torques += nullspace_torques(
                                    env_robots[i].controller.mass_matrix,
                                    nullspace_matrix, 
                                    env_robots[i].controller.initial_joint, 
                                    env_robots[i].controller.joint_pos, 
                                    env_robots[i].controller.joint_vel
        )

        # append joint torques for i-th robot 
        total_torques= np.append(total_torques, torques)

    # delete first element of torques vector 
    total_torques= np.delete(total_torques, 0)

    return total_torques


def desired_pose_to_torque_A(env_robots, desired_pos, desired_ori):
    # to store the total torques
    total_torques= np.array([0])

    # for each robots
    for i in range(2):

        # compute proportional and derivative controller gain
        kp = env_robots[i].controller.nums2array(0.90, 6)
        kd = 2 * np.sqrt(kp) * 1

        # store eef current position and orientation 
        current_pos= env_robots[i].controller.ee_pos
        current_ori= env_robots[i].controller.ee_ori_mat

        # compute position orientation error
        error_pos= desired_pos[i] - current_pos
        error_ori= orientation_error(desired_ori[i], current_ori)

        # compute velocity errors assuming zero velocity goal
        error_vel_pos = - env_robots[i].controller.ee_pos_vel
        error_vel_ori = - env_robots[i].controller.ee_ori_vel

        # compute desired force based on position error and velocity error: F = kp* error_pos + kd* error_vel_pos
        desired_force = np.multiply(np.array(error_pos), np.array(kp[0:3])) + np.multiply(error_vel_pos, kd[0:3])

        # compute desired torque based on orientation error and velocity: tau= kp* error_ori + kd* error_vel_ori
        desired_torque = np.multiply(np.array(error_ori), np.array(kp[3:6])) + np.multiply(error_vel_ori, kd[3:6])

        # calculate relevant control algorithm matrices 
        _, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            env_robots[i].controller.mass_matrix,
            env_robots[i].controller.J_full,
            env_robots[i].controller.J_pos,
            env_robots[i].controller.J_ori
        )

        # decouple desired positional control from ori control
        decoupled_force = np.dot(lambda_pos, desired_force)
        decoupled_torque = np.dot(lambda_ori, desired_torque)
        decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])

        # compute final desired torques without gravity compensation: J^T * F
        torques = np.dot(env_robots[i].controller.J_full.T, decoupled_wrench) 

        # then add nullspace torques
        torques += nullspace_torques(
                                    env_robots[i].controller.mass_matrix,
                                    nullspace_matrix, 
                                    env_robots[i].controller.initial_joint, 
                                    env_robots[i].controller.joint_pos, 
                                    env_robots[i].controller.joint_vel
        )

        # append joint torques for i-th robot 
        total_torques= np.append(total_torques, torques)

    # delete first element of torques vector 
    total_torques= np.delete(total_torques, 0)

    return total_torques
