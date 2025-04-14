import numpy as np
from robosuite.utils.control_utils import *
from robosuite.utils.transform_utils import *

def desired_joint_pos_to_torque(env_robots, desired_qpos0, desired_qpos1):
    
    # to store the total torques
    total_torques= np.array([0])

    # for each robots
    for i in range(2):

        kp= 50
        kp = env_robots[i].controller.nums2array(kp, 7)
        kd = 2 * np.sqrt(kp) * 1

        if i==0: 
            desired_qpos= desired_qpos0
        elif i==1:
            desired_qpos= desired_qpos1

        # torques = pos_err * kp + vel_err * kd
        position_error = desired_qpos - env_robots[i].controller.joint_pos
        vel_pos_error = -env_robots[i].controller.joint_vel

        print("joint position error: ", position_error)
        print("joint velocity error: ", vel_pos_error)

        torques = np.multiply(np.array(position_error), np.array(kp)) + np.multiply(vel_pos_error, kd)

        total_torques= np.append(total_torques, torques)

    # delete first element of torques vector 
    total_torques= np.delete(total_torques, 0)

    return total_torques