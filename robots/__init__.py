from .manipulator import Manipulator
from .single_arm import SingleArm
from .bimanual import Bimanual

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

ROBOT_CLASS_MAPPING = {
    "Baxter": Bimanual,
    "IIWA": SingleArm,
    "Jaco": SingleArm,
    "Kinova3": SingleArm,
    "Panda": SingleArm,
    "Sawyer": SingleArm,
    "UR5e": SingleArm,
    "Panda1": SingleArm,
    "Panda2": SingleArm,
    "Panda0Lemon": SingleArm,
    "Panda1Lemon": SingleArm,
    "Panda0Milk": SingleArm,
    "Panda1Milk": SingleArm,
    "Panda0Can": SingleArm,
    "Panda1Can": SingleArm,
}

BIMANUAL_ROBOTS = {k.lower() for k, v in ROBOT_CLASS_MAPPING.items() if v == Bimanual}
