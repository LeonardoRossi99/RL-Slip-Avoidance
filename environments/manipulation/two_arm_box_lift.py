from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T                             # type: ignore
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv   # type: ignore
from robosuite.models.arenas import TableArena                          # type: ignore
from robosuite.models.objects import BoxObject                          # type: ignore
from robosuite.models.tasks import ManipulationTask                     # type: ignore
from robosuite.utils.observables import Observable, sensor              # type: ignore
from robosuite.utils.placement_samplers import UniformRandomSampler     # type: ignore
import robosuite as suite # type: ignore


class TwoArmPotLift(TwoArmEnv):
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        reward = 0

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0 * direction_coef

        # use a shaping reward
        elif self.reward_shaping:
            # lifting reward
            pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10.0 * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Get contacts
            (g0, g1) = (
                (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
                if self.env_configuration == "bimanual"
                else (self.robots[0].gripper, self.robots[1].gripper)
            )

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            # Grasping reward
            if self._check_grasp(gripper=g0, object_geoms=self.pot.handle0_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            # Grasping reward
            if self._check_grasp(gripper=g1, object_geoms=self.pot.handle1_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward



    def _load_model(self):
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.pot = BoxObject(name="pot")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.pot)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.pot,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.pot,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)
        self.handle0_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle0"])
        self.handle1_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle1"])
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id(self.pot.important_sites["center"])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            if self.env_configuration == "bimanual":
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pf0 = self.robots[0].robot_model.naming_prefix
                pf1 = self.robots[1].robot_model.naming_prefix
            modality = "object"

            # position and rotation of object

            @sensor(modality=modality)
            def pot_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.pot_body_id])

            @sensor(modality=modality)
            def pot_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

            @sensor(modality=modality)
            def handle0_xpos(obs_cache):
                return np.array(self._handle0_xpos)

            @sensor(modality=modality)
            def handle1_xpos(obs_cache):
                return np.array(self._handle1_xpos)

            @sensor(modality=modality)
            def gripper0_to_handle0(obs_cache):
                return (
                    obs_cache["handle0_xpos"] - obs_cache[f"{pf0}eef_pos"]
                    if "handle0_xpos" in obs_cache and f"{pf0}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper1_to_handle1(obs_cache):
                return (
                    obs_cache["handle1_xpos"] - obs_cache[f"{pf1}eef_pos"]
                    if "handle1_xpos" in obs_cache and f"{pf1}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [pot_pos, pot_quat, handle0_xpos, handle1_xpos, gripper0_to_handle0, gripper1_to_handle1]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to each handle
        if vis_settings["grippers"]:
            handles = [self.pot.important_sites[f"handle{i}"] for i in range(2)]
            grippers = (
                [self.robots[0].gripper[arm] for arm in self.robots[0].arms]
                if self.env_configuration == "bimanual"
                else [robot.gripper for robot in self.robots]
            )
            for gripper, handle in zip(grippers, handles):
                self._visualize_gripper_to_target(gripper=gripper, target=handle, target_type="site")

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        return pot_bottom_height > table_height + 0.10

    @property
    def _handle0_xpos(self):
        """
        Grab the position of the left (blue) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle0_site_id]

    @property
    def _handle1_xpos(self):
        """
        Grab the position of the right (green) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle1_site_id]

    @property
    def _pot_quat(self):
        """
        Grab the orientation of the pot body.

        Returns:
            np.array: (x,y,z,w) quaternion of the pot body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

    @property
    def _gripper0_to_handle0(self):
        """
        Calculate vector from the left gripper to the left pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._handle0_xpos - self._eef0_xpos

    @property
    def _gripper1_to_handle1(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._handle1_xpos - self._eef1_xpos
    
