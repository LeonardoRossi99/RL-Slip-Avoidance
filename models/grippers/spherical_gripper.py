from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SphericalGripper(GripperModel):

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/spherical_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "finger": ["sphere_gripper_vis"],
            "corners": [],
            "ft_frame": ["ft_frame"]
        }