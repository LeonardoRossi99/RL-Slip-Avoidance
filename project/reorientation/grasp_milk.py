import numpy as np
from robosuite.project.reorientation.force_control import *
from robosuite.project.reorientation.osc_control import *

def null_action(action_dim):
    return np.zeros(action_dim)

def pre_lift(env, timestep,f0,f1, rlaction):
    
    # lift offset wrt initial position 
    lift_off= np.array([0, 0, 0.2])

    # measure end effectors position and orientation
    eef0_pos= env.robots[0].controller.ee_pos
    eef1_pos= env.robots[1].controller.ee_pos
    eef0_ori= env.robots[0].controller.ee_ori_mat
    eef1_ori= env.robots[1].controller.ee_ori_mat

    # no action
    if timestep< 50:
        action_move = null_action(env.action_dim)
        action_push = null_action(env.action_dim)

    # gras the object 
    if timestep>=50 and timestep<=300:
        null, const= 0.0, 0.1

        const_z = 0.22

        off0 = np.array([null, const,const_z])
        off1 = np.array([null,-const,const_z])
        desired_pos= [env.robots[0].controller.ee_pos + off0, env.robots[1].controller.ee_pos+ off1]
        desired_ori= [env.robots[0].controller.ee_ori_mat, env.robots[1].controller.ee_ori_mat] 
        action_move= desired_pose_to_torque_A(env.robots, desired_pos, desired_ori)
        desired_forces=  np.array([[0, 0, 0.1],[0, 0, 0.1]])
        desired_torques= np.array([[0, 0, 0],[0, 0, 0]])
        action_push= desired_force_to_torque_EEF(env.robots, env._eef0_xmat, env._eef1_xmat, desired_forces, desired_torques)
    
    # lift the object
    elif timestep>300 and timestep<700:
        lift_off = lift_off-np.array([0,0,(timestep-300)*0.0005])
        if timestep== 300:
            eef0_pos= env.robots[0].controller.ee_pos
            eef1_pos= env.robots[1].controller.ee_pos
            eef0_ori= env.robots[0].controller.ee_ori_mat
            eef1_ori= env.robots[1].controller.ee_ori_mat

        desired_pos= [eef0_pos + lift_off, eef1_pos + lift_off]
        desired_ori= [eef0_ori, eef1_ori] 
        action_move= desired_pose_to_torque_A(env.robots, desired_pos, desired_ori)

        desired_forces=  np.array([[0, 0, 0.1],[0, 0, 0.1]])
        desired_torques= np.array([[0, 0, 0],[0, 0, 0]])
        action_push= desired_force_to_torque_EEF(env.robots, env._eef0_xmat, env._eef1_xmat, desired_forces, desired_torques)

    # activate the ReL
    elif timestep>=700:
        action_move= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        action_push = rlaction
    
    return action_move + action_push