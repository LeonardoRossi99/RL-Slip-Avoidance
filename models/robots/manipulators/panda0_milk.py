import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Panda0Milk(ManipulatorModel):
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
            np.arctan2(-0.13714419, 0.99055109),
            np.arctan2(0.95372796,0.30067088),
            np.arctan2(0.23089595,0.97297845),
            np.arctan2( -0.99977765,-0.02108677),
            np.arctan2( 0.99189821,0.12703521),
            np.arctan2( 0.98118109,-0.19308981),
            np.arctan2( 0.88943785,0.45705614)
        ])
    
    
    #('robot0_joint_pos_cos', array([ 0.99488366,  0.40858254,  0.96928887, -0.10336915,  0.17051873,-0.35892365,  0.51666278]))
    # ('robot0_joint_pos_sin', array([-0.10102725,  0.91272138,  0.24592498, -0.99464306,  0.98535443, 0.93336692,  0.85618898]))
     
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
