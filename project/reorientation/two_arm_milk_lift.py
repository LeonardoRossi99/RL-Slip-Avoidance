import numpy as np
import robosuite.utils.transform_utils as T
import math                             
from collections import OrderedDict
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv      
from robosuite.models.arenas import TableArena                              
from robosuite.models.objects import BoxObject
from robosuite.models.objects import CerealObject  
from robosuite.models.objects import CanObject 
from robosuite.models.objects import MilkObject 
from robosuite.models.tasks import ManipulationTask                         
from robosuite.utils.observables import Observable, sensor                 
from robosuite.utils.mjcf_utils import CustomMaterial                       
from robosuite.utils.placement_samplers import UniformRandomSampler         
from robosuite.utils.transform_utils import convert_quat      
from robosuite.project.reorientation.transformations import *                      

class TwoArmMilkLift(TwoArmEnv):
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
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
        control_freq=200,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        off_pos = None,
        target_pos= None,
        target_ang= None,
        force_vec_0= None,
        force_vec_1= None,
        joint_vel0 = None,
        joint_vel1 = None,
        old_dist = None,

        E_p = 0,

        reward_norm_dist = 0,
        reward_velocity = 0,
        reward_terminal_state = 0,
        reward_increment = 0,
        n_slips =0,
        Rel_intervention = 0,
        ReL_intervention_reward = 0,
        n_ReL_act = 0,
        n_ReL_slip_avoid = 0,
        cumulative_ReL_act = [] , 
        cumulative_ReL_slip_avoid = [] ,
        cumulative_slips = []
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.off_pos = off_pos
        self.target_pos= target_pos
        self.target_ang= target_ang
        self.force_vec_0= force_vec_0
        self.force_vec_1= force_vec_1
        self.joint_vel0 = joint_vel0
        self.joint_vel1 = joint_vel1
        self.dist = self.target_pos[2]
        self.old_dist = old_dist
        self.E_p = E_p
        self.reward_norm_dist = reward_norm_dist
        self.reward_velocity = reward_velocity
        self.reward_terminal_state = reward_terminal_state
        self.reward_increment = reward_increment
        self.n_slips = n_slips
        self.Rel_intervention = Rel_intervention
        self.ReL_intervention_reward = ReL_intervention_reward
        self.n_ReL_act = n_ReL_act
        self.n_ReL_slip_avoid = n_ReL_slip_avoid
        self.cumulative_ReL_act = cumulative_ReL_act 
        self.cumulative_ReL_slip_avoid = cumulative_ReL_slip_avoid
        self.cumulative_slips = cumulative_slips

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

    # function to check if
    # the eefs are in contact
    # with the manipulated object
    def check_grasp(self):
        grasp_eef0 = self.check_contact(self.robots[0].gripper.important_geoms["finger"], self.milk)
        grasp_eef1 = self.check_contact(self.robots[1].gripper.important_geoms["finger"], self.milk)
        return grasp_eef0 and grasp_eef1
    
    # function to check if the 
    # object is in contact with table
    def check_contact_table(self):
        table_contact= False
        for contact in self.sim.data.contact:
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if ("table" in geom1 or "table" in geom2) and ("milk_g0" in geom1 or "milk_g0" in geom2):
                table_contact= True
        return table_contact

    # function to compute the reward of
    # the distance between object/target
    def distance_reward_term(self, dist):
        weight = 0.35
        gamma = 50.0
        self.reward_norm_dist= 1/(np.cosh(gamma*dist))
        return weight*self.reward_norm_dist

    # function to compute the reward of
    # the angle distance between object/target
    def angle_reward_term(self, angle_diff):
        weight = 0.45
        gamma = 0.5/5
        reward_angle = 1/(np.cosh(gamma*angle_diff))
        return weight*reward_angle

    # function to compute the reward associated 
    # with the reaching of the target height 
    def terminal_state_reward_term(self, dist):
        weight = 0.0
        gamma_t = 100.0
        self.reward_terminal_state = 1/(np.cosh(gamma_t*dist))
        return weight*self.reward_terminal_state
    
    def terminal_state_ang_reward_term(self, ang):
        weight = 1
        return weight*1

    # function to compute the reward to encourage
    # approach the target height with small velocity
    def velocity_reward_term(self, dist, vel):
        weight = 0.0
        gamma_d = 100.0
        gamma_v = 5.0
        self.reward_velocity = (1/np.cosh(gamma_d*dist))*(1/np.cosh(gamma_v*vel))
        return weight*self.reward_velocity
    
    # function to compute the pseudo
    # energy term based on residuals
    def pseudo_energy_term(self):
        reward= self.Rel_intervention
        self.ReL_intervention_reward = reward
        return reward 

    def reward(self, action= None):
        
        # total reward
        reward = 0

        # convex combination
        # cost coefficient 
        
        # 1-2: set to zero
        #Lambda = 1

        # set to one
        Lambda = 0.8

        # define the intervals within the associated reward terms are activated
        interval_velocity_term = [-0.3 * self.off_pos[2], +0.3 * self.off_pos[2]]
        interval_terminal_term = [-0.1 * self.off_pos[2], +0.1 * self.off_pos[2]]

        # init reward terms to plot
        self.reward_norm_dist = 0
        self.reward_velocity = 0
        self.reward_terminal_state = 0

        if self.reward_shaping:

            # rewards term (not penalty) computed if grasping and
            # not object-talbe touching conditions are satisfied 
            if self.check_grasp() and not self.check_contact_table():

                # compute the normalized distance between current position and the target position
                dist = self.target_pos[2] - self.sim.data.body_xpos[self.milk_body_id][2]

                # compute the linear velocity of robot end effector and divide it by the number of joints
                vel = 100*np.linalg.norm(np.dot(self.robots[0].controller.J_full, self.joint_vel0)[:3])/7

                # compute the angle difference between the current angle and the target one
                obj_z_axis_world = np.matmul(quat2mat(convert_quat(np.array(self.sim.data.body_xquat[self.milk_body_id]), to="xyzw")), np.array([0, 0, 1]))
                angle_diff = rotation_difference_x(obj_z_axis_world, self.target_ang)
                
                # 1: DISTANCE-based reward function term
                reward += self.distance_reward_term(dist)

                # 1.1: ANGLE-based reward function term
                reward += self.angle_reward_term(angle_diff)

                # 2: VELOCITY-based reward function term
                if  interval_velocity_term[0] <= dist < interval_velocity_term[1]:
                    reward += self.velocity_reward_term(dist, vel)

                # 3: TERMINAL STATE-based reward function term
                if interval_terminal_term[0]<= dist< interval_terminal_term[1]:
                    reward += self.terminal_state_reward_term(dist)

                # 3.1: TERMINAL ANGLE-based reward function term
                if -0.5 <=angle_diff<= 0.5:
                    reward += 1

            # 4: PENALITY: object not grasped
            if not self.check_grasp():
                reward -= 1

            # 5: PENALITY: object touch table
            if self.check_contact_table():
                reward -= 0.5

            # 6: PSEUDO ENERGY TERM based on KF residual
            reward_PS = self.pseudo_energy_term()

            self.E_p = reward

            # general REWARD FUNCTION: convex combination
            reward = Lambda*reward+ (1-Lambda) * reward_PS

        # scale the total reward term
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward




    def _load_model(self):

        super()._load_model()
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {
            "type": "milk",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        milk = CustomMaterial(
            texture="Ceramic",
            tex_name="milk",
            mat_name="milk_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.milk = MilkObject(
            name="milk",
            #size_min=[0.050, 0.050, 0.050],
            #size_max=[0.052, 0.052, 0.052],
            #rgba=[1, 0, 0, 1],
            #material=milk,
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.milk)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.milk,
                x_range=[0.00, 0.00],
                y_range=[0.00, 0.00],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rotation=0,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.milk,
        )




    def _setup_references(self):
        super()._setup_references()
        self.milk_body_id = self.sim.model.body_name2id(self.milk.root_body)



    def _setup_observables(self):

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

            # milk-related observables
            @sensor(modality=modality)
            def milk_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.milk_body_id])

            @sensor(modality=modality)
            def milk_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.milk_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper0_to_milk_pos(obs_cache):
                return (
                    obs_cache[f"{pf0}eef_pos"] - obs_cache["milk_pos"]
                    if f"{pf0}eef_pos" in obs_cache and "milk_pos" in obs_cache
                    else np.zeros(3)
                )
            
            @sensor(modality=modality)
            def gripper1_to_milk_pos(obs_cache):
                return (
                    obs_cache[f"{pf1}eef_pos"] - obs_cache["milk_pos"]
                    if f"{pf1}eef_pos" in obs_cache and "milk_pos" in obs_cache
                    else np.zeros(3)
                )
            
            @sensor(modality=modality)
            def gripper0_contact_force(obs_cache):
                return self.robots[0].ee_force
            
            @sensor(modality=modality)
            def gripper0_contact_torque(obs_cache):
                return self.robots[0].ee_torque
            
            @sensor(modality=modality)
            def gripper1_contact_force(obs_cache):
                return self.robots[1].ee_force
            
            @sensor(modality=modality)
            def gripper1_contact_torque(obs_cache):
                return self.robots[1].ee_torque

            sensors = [milk_pos,
                       milk_quat,
                       gripper0_to_milk_pos,
                       gripper1_to_milk_pos,
                       gripper0_contact_force,
                       gripper0_contact_torque,
                       gripper1_contact_force,
                       gripper1_contact_torque
                       ]
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

        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))




    def visualize(self, vis_settings):
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the milk
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.milk)


    def _check_success(self):

        # compute the position of the milk
        milk_pos = self.sim.data.body_xpos[self.milk_body_id]                           
            
        # compute the positions of robot0 and robot1 grippers
        gripper0_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        gripper1_site_pos = self.sim.data.site_xpos[self.robots[1].eef_site_id]
        dist0 = np.linalg.norm(gripper0_site_pos - milk_pos)
        dist1 = np.linalg.norm(gripper1_site_pos- milk_pos)
        dist= dist0 + dist1

        return dist < 0.1

        #milk_height = self.sim.data.body_xpos[self.milk_body_id][2]
        #table_height = self.model.mujoco_arena.table_offset[2]
        #return milk_height > table_height + 0.04



    @property
    def _handle0_xpos(self):
        return self.sim.data.site_xpos[self.handle0_site_id]

    @property
    def _handle1_xpos(self):
        return self.sim.data.site_xpos[self.handle1_site_id]