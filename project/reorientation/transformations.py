import numpy as np
from robosuite.utils.transform_utils import *

def rotation_difference_x(v1, v2):
    
    v1_yz = np.array([v1[1], v1[2]])
    v2_yz = np.array([v2[1], v2[2]])
    
    norm_v1 = np.linalg.norm(v1_yz)
    norm_v2 = np.linalg.norm(v2_yz)
    
    dot_product = np.dot(v1_yz, v2_yz)

    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    
    sign = np.cross([0, v1_yz[0], v1_yz[1]], [0, v2_yz[0], v2_yz[1]])[0]
    if sign < 0:
        angle_deg = -angle_deg 
    
    return angle_deg




def orientation_difference(v1, v2):
    """
    Function to compute the orientation difference 
    between two vectors
    """
    return np.arccos(np.dot(v1, v2)/ (np.linalg.norm(v1)*np.linalg.norm(v2)))


def contact_angle(eef_quat, obj_quat):
    """
    Function to compute the angle between the normal
    to the object and the actuation force direction

    Args:
        eef_quat: robot end effector quaternion
        obj_quat: pickable object quaternion
    """

    eef_R_mat= quat2mat(eef_quat)
    obj_R_mat= quat2mat(obj_quat)

    z_eef= eef_R_mat[:,2]
    z_obj = obj_R_mat[:,2]

    ori_error_z_eef_obj= orientation_difference(z_eef, z_obj)
    theta_rad= ori_error_z_eef_obj - np.pi/2

    return theta_rad
