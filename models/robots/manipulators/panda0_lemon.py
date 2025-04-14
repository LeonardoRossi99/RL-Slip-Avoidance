import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Panda0Lemon(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array([
                np.arctan2(-0.09506867, 0.99547072)-0.01,
                np.arctan2(0.94220144,0.33504692),
                np.arctan2(0.25639838,0.96657119),
                np.arctan2(-0.99842455,-0.05611078),
                np.arctan2(  0.98505553,0.17223708),
                np.arctan2(  0.92548083,-0.37879444),
                np.arctan2(  0.84129219, 0.54058066)
                ])
    
    #robot0_joint_pos_cos', array([ 0.99547072,  0.33504692,  0.96657119, -0.05611078,  0.17223708,-0.37879444,  0.54058066])),
    #robot0_joint_pos_sin', array([-0.09506867,  0.94220144,  0.25639838, -0.99842455,  0.98505553,0.92548083,  0.84129219])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
