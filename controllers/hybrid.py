import math
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *

class HybridOscTorqueControl(Controller):

 
    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        kp=150,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        torque_limits=None,
        interpolator=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_ori=True,
        control_delta=True,
        uncouple_pos_ori=True,
        **kwargs,
    ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # set all parameters for operations space control(osc) 
        self.use_ori = control_ori      # is set True as default
        self.use_delta = control_delta  # is set True as default
        self.impedance_mode = impedance_mode
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori
        self.uncoupling = uncouple_pos_ori
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)
        self.relative_ori = np.zeros(3)
        self.ori_ref = None


        # set all parameters for joint torque control
        self.control_dim = len(joint_indexes["joints"])
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)
        self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits
        self.control_freq = policy_freq
        self.interpolator = interpolator
        self.goal_torque = None
        self.current_torque = np.zeros(self.control_dim)
        self.torques = None



    def set_goal(self, torques):

        self.update()
        assert len(torques) == self.control_dim, "Delta torque must be equal to the robot's joint dimension space!"
        self.goal_torque = np.clip(self.scale_action(torques), self.torque_limits[0], self.torque_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)



    def set_goal(self, action, set_pos=None, set_ori=None):

        self.update()                               # case self.impedence is fixed
        delta = action                              # case interpret actions as deltas
        scaled_delta = self.scale_action(delta)    
        bools = [0.0 if math.isclose(elem, 0.0) else 1.0 for elem in scaled_delta[3:]]
        if sum(bools) > 0.0 or set_ori is not None:
            self.goal_ori = set_goal_orientation(
                scaled_delta[3:], self.ee_ori_mat, orientation_limit=self.orientation_limits, set_ori=set_ori
            )
        self.goal_pos = set_goal_position(
            scaled_delta[:3], self.ee_pos, position_limit=self.position_limits, set_pos=set_pos
        )

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )
            self.relative_ori = np.zeros(3)




    def run_controller(self):

        if self.goal_torque is None:
            self.set_goal(np.zeros(self.control_dim))

        self.update()

        if self.interpolator is not None:
            if self.interpolator.order == 1:
                self.current_torque = self.interpolator.get_interpolated_goal()
            else:
                pass
        else:
            self.current_torque = np.array(self.goal_torque)

        self.torques = self.current_torque + self.torque_compensation

        super().run_controller()
        return self.torques
    



    def run_controller(self):

        # Update state
        self.update()

        desired_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_pos = np.array(self.goal_pos)

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        # Compute desired force and torque based on errors
        position_error = desired_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel

        # F_r = kp * pos_err + kd * vel_err
        desired_force = np.multiply(np.array(position_error), np.array(self.kp[0:3])) + np.multiply(
            vel_pos_error, self.kd[0:3]
        )

        vel_ori_error = -self.ee_ori_vel

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(np.array(ori_error), np.array(self.kp[3:6])) + np.multiply(
            vel_ori_error, self.kd[3:6]
        )

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques
    



    def reset_goal(self):

        self.goal_torque = np.zeros(self.control_dim)
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    @property
    def name(self):
        return "HYBRID_OSC_JOINT_TORQUE"