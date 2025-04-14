import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Panda1Lemon(ManipulatorModel):
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
                np.arctan2(0.11152408, 0.99376173)+0.01,
                np.arctan2(0.95082676, 0.30972321),
                np.arctan2(-0.25423274,0.96714307),
                np.arctan2(-0.99952255,-0.03089769),
                np.arctan2(-0.97766863,0.21015243),
                np.arctan2(0.96538259,-0.26083797),
                np.arctan2( 0.82905721,0.55916379)
                ])
    
    #robot1_joint_pos_cos', array([ 0.99376173,  0.30972321,  0.96714307, -0.03089769,  0.21015243,  -0.26083797,  0.55916379])),
    # robot1_joint_pos_sin', array([ 0.11152408,  0.95082676, -0.25423274, -0.99952255, -0.97766863,0.96538259,  0.82905721]))


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
