import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Panda2(ManipulatorModel):
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
                np.arctan2(0, 0)+0.01,
                np.arctan2(0.0),
                np.arctan2(0,0),
                np.arctan2(0,0),
                np.arctan2(  0,0),
                np.arctan2(0,0),
                np.arctan2( 0,0)
                ])
    """print("")
    print("panda 2 joint 1: ", np.arctan2(0.11404043, 0.99347611)+0.01)*(180/np.pi)
    print("panda 2 joint 2: ", np.arctan2(0.9114359,0.41144211)*(180/np.pi))
    print("panda 2 joint 3: ", np.arctan2(-0.24553737,0.96938713)*(180/np.pi))
    print("panda 2 joint 4: ", np.arctan2(-0.99403106,-0.10909741)*(180/np.pi))
    print("panda 2 joint 5: ", np.arctan2(-0.98100988,0.19395776)*(180/np.pi))
    print("panda 2 joint 6: ", np.arctan2(0.97162295,-0.23653507)*(180/np.pi))
    print("panda 2 joint 7: ", np.arctan2(0.81641223,0.57746954)*(180/np.pi))"""
    


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
