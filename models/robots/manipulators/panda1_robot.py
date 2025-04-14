import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Panda1(ManipulatorModel):
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
                np.arctan2(-0.09521214, 0.995457)-0.01,
                np.arctan2(0.88378991,0.46788395),
                np.arctan2(0.24735157,0.9689258),
                np.arctan2(-0.9836927,-0.17985737),
                np.arctan2(  0.98963127,0.14363133),
                np.arctan2(  0.93261121,-0.36088272),
                np.arctan2(  0.84819719, 0.5296806)
                ])
    
    """print("panda 1 joint 1: ", (np.arctan2(-0.09521214, 0.995457)-0.01)*(180/np.pi))
    print("panda 1 joint 2: ", np.arctan2(88378991, 0.46788395)*(180/np.pi))
    print("panda 1 joint 3: ", np.arctan2(0.24735157, 0.9689258)*(180/np.pi))
    print("panda 1 joint 4: ", np.arctan2(-0.9836927, -0.17985737)*(180/np.pi))
    print("panda 1 joint 5: ", np.arctan2(0.98963127, 0.14363133)*(180/np.pi))
    print("panda 1 joint 6: ", np.arctan2(0.93261121, -0.36088272)*(180/np.pi))
    print("panda 1 joint 7: ", np.arctan2(0.84819719, 0.5296806)*(180/np.pi))"""


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
