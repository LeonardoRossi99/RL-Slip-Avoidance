import sys
import math
import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from robosuite.utils.control_utils import *
from robosuite.utils.transform_utils import *
from robosuite.controllers import *
from robosuite.project.lift import two_arm_can_lift
from robosuite.project.lift.transformations import * 
from robosuite.project.lift.env_set import *
from robosuite.project.lift.grasp_can import *
from robosuite.project.lift.reactive_layer_can import *
from robosuite.project.lift.measurements_layer_can import *
from robosuite.project.lift.force_control import *

# True  if training process
# False if test
training = False

# environment parameters
control_freq=150
t_max = 2500
t_pre = 800
t_measure = 45

# static control parameters
mu= 0.9
k= 1.1

# dynamic control parameters
T = 1/control_freq
K = 2500
ki= 8
F_ = np.array([[1, T], [0, 1]])
H = np.array([[K, 0]])
Q = np.array([[1, 0], [0, 1]])
R = np.array([[1]])
X0 = np.array([[0],[0]])
P0 = Q


# offset taget height with respect to
# the initial object position at start
offs_pos = np.array([0.0, 0.0, 0.03])

class Agent():

    # function to inititialize the Agent structure 
    def __init__(self, env, input_size, output_size):
        self.env = env
        self.input_size = input_size
        self.output_size = output_size
        self.A = np.zeros(output_size)
        self.b = np.zeros(output_size)
        self.MeL = MeasurementsLayer()
        self.ReL = ReactiveLayer(mu, k, F_, H, Q, R, X0, P0, ki, T)
        
    # function to compute the system state
    def get_state(self):
        return np.array([env.target_pos[2] - env.sim.data.body_xpos[env.can_body_id][2]])

    # function to set the policy weights
    def set_weights(self, weights):
        w = 0
        for i in range(self.output_size):
            self.A[i] = weights[w]
            self.b[i] = weights[w+self.output_size]
            w+=1
    
    # function to compute the policy weights dimension 
    def get_weights_dim(self):
        return 2*self.output_size
    
    # function to compute the policy action
    def forward(self, x):
        dist_normalized = min(1, x/offs_pos[2])
        dist_mormalized_modified = 1-(1-dist_normalized)**2
        return self.A*dist_mormalized_modified + self.b

    # function to evaluate the performance based on total return of a given policy parametrization defined by its weights
    def evaluate(self, weights, max_n_timesteps=t_max, gamma=1.0, pre_lift_timesteps= t_pre, timestep_to_measure=t_measure):
        
        # initialize Agent measurament layer, state, weights
        self.MeL.set_time_to_measure(timestep_to_measure)
        obs= env.reset()
        state = self.get_state()
        self.set_weights(weights)
    
        # initialize epsiode parameters
        episode_return = 0.0
        episode_ReLint = 0.0
        full_actuation_t= 250
        correctly_lift_t = 0
        correctly_lift_time = 250
        alreadyLifted = False
        alreadyTouchGround = False
        alreadyPlacedOverTheTable = False
        E_p = 0.0
        
        # initialize intervention coefficients
        csi = 0.0
        tau = 1.0 

        # for each epsiode time-step do:
        for t in range(max_n_timesteps):

            # estimate finger normal actuation force componenr
            nd0, nd1 = agent.MeL.get_actuation_dir(obs)
            
            # measure tangential force normalized wrt initial ones
            tg0, tg1 = self.MeL.measure_tangent_forces(t, obs)

            # measure each robot joints velocity 
            env.joint_vel0 = obs["robot0_joint_vel"]
            env.joint_vel1 = obs["robot1_joint_vel"]

            # execute pre lifting action
            if t<pre_lift_timesteps:
                ReL_action, _, _, _,_,_= self.ReL.action(env.robots, env._eef0_xmat, env._eef1_xmat, tg0, tg1, nd0, nd1)
                action = pre_lift(env, t, None, None, ReL_action)
                obs, reward, done, _,= self.env.step(action)
            
            else:
                # compute the agent state 
                state = self.get_state()
                
                # set initial object position
                if t== pre_lift_timesteps:
                    env.target_pos = np.array(env.sim.data.body_xpos[env.can_body_id]) + offs_pos
                    env.Rel_intervention = 0
                    
                # compute the agent action
                force = self.forward(state)
                Agent_action = force2torques(env, force)
                ReL_action, _, _ ,_,e0,e1= self.ReL.action(env.robots, env._eef0_xmat, env._eef1_xmat, tg0, tg1, nd0, nd1)
                
                obs, reward, done, _, = self.env.step(csi*Agent_action + tau*ReL_action)
                #obs, reward, done, _, = self.env.step(Agent_action + ReL_action)


                # increase the Agent Linear Policy intervention 
                if pre_lift_timesteps< t <pre_lift_timesteps+ full_actuation_t:
                    csi += 0.004

                # update intervention parameters
                if t>=pre_lift_timesteps:
                    csi = min(1/(np.cosh(80*max(np.abs(e0[0][0]), np.abs(e1[0][0]))))+0.00001, csi)
                    tau = 1 + (1-csi)/20
                    env.Rel_intervention = csi
                    episode_ReLint += (1-csi)/(max_n_timesteps- pre_lift_timesteps -full_actuation_t)

                # check if the object is correctly lifted 
                if not alreadyLifted and -0.1*offs_pos[2] <=state<= +0.1*offs_pos[2]:
                    correctly_lift_t += 1
                    if correctly_lift_t == correctly_lift_time:
                            print("Lift")
                            alreadyLifted = True
                
                # check if the object has been slipped by checking if it touch the table
                if  t>=500 and not alreadyPlacedOverTheTable and not alreadyTouchGround:
                    if env.check_contact_table() and not env.check_grasp():
                        alreadyTouchGround = True
                        print("Touch the ground")

                # check if the object has been placed over the table by checking if it is grasped and touch the table
                if (t>=500) and (env.check_contact_table()) and (env.check_grasp()) and (not alreadyPlacedOverTheTable):
                    print("Object placed over the table")
                    alreadyPlacedOverTheTable=True

                # compute the discounted total episode return
                episode_return += reward * math.pow(gamma, t)
                E_p += env.E_p

                if done:
                    break

        if alreadyTouchGround: 
            env.n_slips +=1

        return episode_return, episode_ReLint, alreadyLifted, E_p
    
    def evaluate_(self, weights, max_n_timesteps=t_max, gamma=1.0, pre_lift_timesteps= t_pre, timestep_to_measure=t_measure):
        
        pos_vec = np.array([0])
        vec_dynamic_correction = np.array([0])

        # initialize Agent measurament layer, state, weights
        self.MeL.set_time_to_measure(timestep_to_measure)
        obs= env.reset()
        state = self.get_state()
        self.set_weights(weights)
    
        # initialize epsiode parameters
        episode_return = 0.0
        episode_ReLint = 0.0
        full_actuation_t= 250
        correctly_lift_t = 0
        correctly_lift_time = 250
        alreadyLifted = False
        alreadyTouchGround = False
        alreadyPlacedOverTheTable = False
        E_p = 0.0
        
        # initialize intervention coefficients
        csi = 0.0
        tau = 1.0 

        # for each epsiode time-step do:
        for t in range(max_n_timesteps):

            # estimate finger normal actuation force componenr
            nd0, nd1 = agent.MeL.get_actuation_dir(obs)
            
            # measure tangential force normalized wrt initial ones
            tg0, tg1 = self.MeL.measure_tangent_forces(t, obs)

            # measure each robot joints velocity 
            env.joint_vel0 = obs["robot0_joint_vel"]
            env.joint_vel1 = obs["robot1_joint_vel"]

            # execute pre lifting action
            if t<pre_lift_timesteps:
                ReL_action, _, _, _,_,_= self.ReL.action(env.robots, env._eef0_xmat, env._eef1_xmat, tg0, tg1, nd0, nd1)
                action = pre_lift(env, t, None, None, ReL_action)
                obs, reward, done, _,= self.env.step(action)
            
            else:
                # compute the agent state 
                state = self.get_state()
                pos_vec = np.append(pos_vec, env.sim.data.body_xpos[env.can_body_id][2])
                

                # set initial object position
                if t== pre_lift_timesteps:
                    env.target_pos = np.array(env.sim.data.body_xpos[env.can_body_id]) + offs_pos
                    env.Rel_intervention = 0
                    
                # compute the agent action
                force = self.forward(state)
                Agent_action = force2torques(env, force)
                ReL_action, _, _ ,_,e0,e1= self.ReL.action(env.robots, env._eef0_xmat, env._eef1_xmat, tg0, tg1, nd0, nd1)
                obs, reward, done, _, = self.env.step(csi*Agent_action + tau*ReL_action)

                # increase the Agent Linear Policy intervention 
                if pre_lift_timesteps< t <pre_lift_timesteps+ full_actuation_t:
                    csi += 0.004

                # update intervention parameters
                if t>=pre_lift_timesteps:
                    csi = min(1/(np.cosh(80*max(np.abs(e0[0][0]), np.abs(e1[0][0]))))+0.00001, csi)
                    tau = 1 + (1-csi)/20
                    env.Rel_intervention = csi
                    episode_ReLint += (1-csi)/(max_n_timesteps- pre_lift_timesteps -full_actuation_t)

                vec_dynamic_correction = np.append(vec_dynamic_correction, max(np.abs(e0[0][0]), np.abs(e1[0][0])))

                # check if the object is correctly lifted 
                if not alreadyLifted and -0.1*offs_pos[2] <=state<= +0.1*offs_pos[2]:
                    correctly_lift_t += 1
                    if correctly_lift_t == correctly_lift_time:
                            print("Lift")
                            alreadyLifted = True
                
                # check if the object has been slipped by checking if it touch the table
                if  t>=500 and not alreadyPlacedOverTheTable and not alreadyTouchGround:
                    if env.check_contact_table() and not env.check_grasp():
                        alreadyTouchGround = True
                        print("Touch the ground")

                # check if the object has been placed over the table by checking if it is grasped and touch the table
                if (t>=500) and (env.check_contact_table()) and (env.check_grasp()) and (not alreadyPlacedOverTheTable):
                    print("Object placed over the table")
                    alreadyPlacedOverTheTable=True

                # compute the discounted total episode return
                episode_return += reward * math.pow(gamma, t)
                E_p += env.E_p

                if done:
                    break

        if alreadyTouchGround: 
            env.n_slips +=1
        
        pos_vec = np.delete(pos_vec, 0)
        vec_dynamic_correction = np.delete(vec_dynamic_correction, 0)

        return episode_return, episode_ReLint, alreadyLifted, E_p, pos_vec, vec_dynamic_correction


# function to compute the weighted mean given a vector
# of weights and the associated vector of weights-score
def weighted_mean(elite_weights, elite_weights_scores):
    elite_weights = np.array(elite_weights) 
    elite_weights_scores = np.array(elite_weights_scores)
    elite_weights_scores_normalized = elite_weights_scores / np.sum(elite_weights_scores)
    best_weight_new = np.average(elite_weights, axis=0, weights=elite_weights_scores_normalized)
    return best_weight_new

# function to compute the Cross Entropy Method CEM for a given maximum of iterations, episode time horizon, and algorithm parametrization
def CEM(n_training_iterations=25, max_n_timesteps=2500, gamma=1.0, pop_size=15, n_elite=3, sigma=0.004, alpha= 0.95, beta= 0.2):
    LIFT = False
    alreadyLIFT= False
    firstLift = n_training_iterations
    best_score = []
    mean_score = []
    E_p_best_score = []
    slip_number = []
    intervent_score = []
    best_score_intervent = []
    E_ps = []
    
    # initialize the mean distribution as best weight 
    mean_weight = np.zeros(agent.get_weights_dim())
    best_weight = mean_weight

    # for each episode until the robot lifts the object
    for i_iteration in range(1, n_training_iterations+1):
        print("Episode: ", i_iteration)
        rewards = []
        intervents = []

        # generate new weights population by adding to the last iteration mean-weight noise from standard deviation
        weights_pop = [mean_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size-1)]        
        weights_pop.append(best_weight)
        
        # evaluate each weight
        for i in range(len(weights_pop)):
            reward, intervent, isLifted, ep = agent.evaluate(weights_pop[i], max_n_timesteps, gamma)
            rewards.append(reward)
            E_ps.append(ep)
            intervents.append(intervent)
            print(f"return {i}: {reward}")
            print("")
            if LIFT==False and isLifted:
                LIFT=True
        
        if LIFT and not alreadyLIFT:
            firstLift= i_iteration
            alreadyLIFT= True

        # selcect the elite weights based on total return
        elite_idxs = np.array(rewards).argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        elite_weights_scores = [np.array(rewards)[i] for i in elite_idxs]
        best_weight = elite_weights[n_elite-1]
        best_reward = elite_weights_scores[n_elite-1]
        best_reward_intervent = intervents[elite_idxs[n_elite-1]]
        print("Elite indexes: ", elite_idxs)

        # update the mean and covariance of the normal distribution
        mean_weight_new = weighted_mean(elite_weights, elite_weights_scores)
        mean_weight= alpha*mean_weight_new + (1-alpha)*mean_weight
        sigma_new = np.array(elite_weights).std(axis=0)
        sigma= beta*sigma_new + (1-beta)*sigma
        
        # evaluate return of the mean weight and check if it the best
        mean_reward,_, _, _= agent.evaluate(mean_weight, gamma=1.0)
        if mean_reward >= best_reward:
            best_weight = mean_weight
        
        # resize the population set
        # and the elite set dimension
        pop_size = max(8, pop_size-1)
        n_elite  = max(2, math.ceil(pop_size/5))

        # update scores list to plot later 
        mean_score.append(mean_reward)
        best_score.append(best_reward)
        slip_number.append(env.n_slips)
        E_p_best_score.append(E_ps[elite_idxs[n_elite-1]])
        print("Pose Term", E_ps[elite_idxs[n_elite-1]])
        intervent_score.append(np.mean(np.array(intervents), axis=0))
        best_score_intervent.append(best_reward_intervent)

        # save the policy weights parametrization
        if i_iteration == n_training_iterations:
            np.savetxt('Lift_checkpoint.txt', best_weight)
            _,_,_,_, pos_vec, vec_dynamic_correction = agent.evaluate_(mean_weight, gamma=1.0)

    return mean_score, best_score, slip_number, intervent_score, best_score_intervent, firstLift, E_p_best_score, pos_vec, vec_dynamic_correction



# define environment type
usr_env_type = sys.argv[1]
env_params = set_environment(usr_env_type)

# define environment parameters
env_name = env_params[0]
env_first_robot = env_params[1]
env_second_robot = env_params[2]
env_controller = env_params[3]
env_gripper = env_params[4]
env_arm_conf = env_params[5]
env_obj_pos = env_params[6]
env_obj_quat = env_params[12]

# define the environment controller configuration as joint torque controller
controller_config_ = load_controller_config(default_controller=env_controller)

# set environment
env = suite.make(
    env_name=env_name,
    robots=[env_first_robot, env_second_robot],
    gripper_types=env_gripper,
    controller_configs=controller_config_,
    env_configuration=env_arm_conf, 
    has_renderer=True,                      
    has_offscreen_renderer=False,  
    reward_shaping= True,         
    ignore_done=True,
    control_freq=control_freq,
    horizon=200,
    use_camera_obs=False,
    off_pos = offs_pos,
    target_pos= np.array([0.0, 0.0, 0.0]),
    target_ang= np.array([0.0, 0.0, 0.0]),
    force_vec_0= np.array([0,0,0]),
    force_vec_1= np.array([0,0,0]),
    joint_vel0 = None,
    joint_vel1 = None,
    old_dist = 0,
    E_p = 0,
    reward_norm_dist = 0,
    reward_velocity = 0,
    reward_terminal_state = 0,
    reward_increment = 0, 
    n_slips = 0,
    Rel_intervention = 0,
    ReL_intervention_reward = 0,
    n_ReL_act = 0,
    n_ReL_slip_avoid = 0,
    cumulative_ReL_act = [] , 
    cumulative_ReL_slip_avoid = [] ,
    cumulative_slips = []
)

# modify environment simulation frequency
env.sim.model.opt.timestep = 0.002

# initialize environment
obs= env.reset()
done = False
ret = 0.0

# instantiate agent object
agent = Agent(env, 1, 6)


# training
if training:

    # apply CEM and return the total return score, number of slipping and evaluation of
    # reactive layer intervention along each episode and the one associated to the best score
    mean_score, best_scores, cumulative_slips, intervent_score, best_score_intervent, i_firstlift, E_p, pos_vec, vec_dynamic_correction = CEM()

    np.save('mean_score.npy', mean_score)
    np.save('best_scores.npy', best_scores)
    np.save('cumulative_slips.npy', cumulative_slips)
    np.save('intervent_score.npy', intervent_score)
    np.save('best_score_intervent.npy', best_score_intervent) 
    np.save('E_p.npy', E_p) 
    np.save('pos_vec.npy', pos_vec)
    np.save('vec_dynamic_correction.npy', vec_dynamic_correction)

    # set important evaluation plot
    plt.figure(1, figsize=(10, 6))
    plt.plot(np.arange(1, len(mean_score) + 1), mean_score, label="Mean weight Return")
    plt.plot(np.arange(1, len(best_scores) + 1), best_scores, label="Best weight Return", linestyle="--", color="b")
    plt.axvline(i_firstlift, color='r', linestyle=':', linewidth=2)
    plt.title("Total Return Across Episodes")
    plt.ylabel("Return")
    plt.xlabel("Episode #")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(2, figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_slips) + 1), cumulative_slips, label="Number of times the object slips", color="b")
    plt.title("Number of times the object slip across all episodes")
    plt.ylabel("Slips #")
    plt.xlabel("Episode #")
    plt.grid(True)
    plt.show()

    plt.figure(3, figsize=(10, 6))
    plt.plot(np.arange(1, len(intervent_score) + 1), intervent_score)
    plt.plot(np.arange(1, len(best_score_intervent) + 1), best_score_intervent, color="b")
    plt.title("Reactive Layer Mean Intervent Score for Each Episode")
    plt.ylabel("Intervention evaluation")
    plt.xlabel("Episode #")
    plt.grid(True)
    plt.show()

    plt.figure(4, figsize=(10, 6))
    plt.plot(np.arange(1, len(pos_vec) + 1), pos_vec)
    plt.title("Object Position")
    plt.ylabel("Object Position")
    plt.xlabel("Episode #")
    plt.grid(True)
    plt.show()

    plt.figure(4, figsize=(10, 6))
    plt.plot(np.arange(1, len(vec_dynamic_correction) + 1), vec_dynamic_correction)
    plt.title("vec_dynamic_correction")
    plt.ylabel("vec_dynamic_correction")
    plt.xlabel("Episode #")
    plt.grid(True)
    plt.show()





else:
    # set the agent and the environment and run simulation
    agent.set_weights(np.loadtxt("Lift_checkpoint.txt"))
    obs = env.reset()
    state= agent.get_state()
    init_pos= None
    timestep_to_measure= 45
    MeL = MeasurementsLayer()
    ReL = ReactiveLayer(mu, k, F_, H, Q, R, X0, P0, ki, T)
    episode_return = 0.0
    t=0
    dist_vec= np.array([0])
    dist_vecx= np.array([0])
    dist_vecy= np.array([0])
    dist_vecz= np.array([0])
    pos_vec = np.array([0])
    reward_vec = np.array([0])
    reward_norm_dist_vec = np.array([0])
    reward_velocity_vec = np.array([0])
    reward_terminal_state_vec = np.array([0])
    reward_increment_vec = np.array([0])
    reward_ReL_intervention_vec = np.array([0])
    tangential_force0 = np.array([0])
    tangential_force1 = np.array([0])
    residual_filter0 = np.array([0])
    residual_filter1 = np.array([0])

    t_vec = np.array([0])
    t_vec_tot = np.array([0])

    alreadyRefelxHanging = False
    alreadyLifted = False
    agent.MeL.set_time_to_measure(t_measure)
    obs= env.reset()
    csi = 0.004
    tau = 1

    target_z_axis_world = None

    for t in range(t_max):

        nd0, nd1 = agent.MeL.get_actuation_dir(obs)
        tg0, tg1 = agent.MeL.measure_tangent_forces(t, obs)
        env.joint_vel0 = obs["robot0_joint_vel"]
        env.joint_vel1 = obs["robot1_joint_vel"]
        #print(t)
        
        if t<t_pre:
            ReL_action ,_, _, Rel_intervention, e0, e1= agent.ReL.action(env.robots, env._eef0_xmat, env._eef1_xmat, tg0, tg1, nd0, nd1)
            action = pre_lift(env, t, None, None, ReL_action)
            obs, reward, done, _,= agent.env.step(action)
            #env.render()
            init_pos = np.array(env.sim.data.body_xpos[env.can_body_id])
            residual_filter0 = np.append(residual_filter0, e0)
            residual_filter1 = np.append(residual_filter1, e1)
            tangential_force0 = np.append(tangential_force0, tg0)
            tangential_force1 = np.append(tangential_force1, tg1)
            t_vec_tot = np.append(t_vec_tot, t)

        else:
            state = agent.get_state()
            if t==t_pre:
                env.target_pos = init_pos + offs_pos
                env.Rel_intervention = 0

            dist= np.linalg.norm(env.sim.data.body_xpos[env.can_body_id] - env.target_pos)
            distx= np.linalg.norm(env.sim.data.body_xpos[env.can_body_id][0] - env.target_pos[0])
            disty= np.linalg.norm(env.sim.data.body_xpos[env.can_body_id][1] - env.target_pos[1])
            distz= np.linalg.norm(env.sim.data.body_xpos[env.can_body_id][2] - env.target_pos[2])
            pos_vec = np.append(pos_vec, env.sim.data.body_xpos[env.can_body_id][2])
            dist_vec= np.append(dist_vec, dist)
            t_vec = np.append(t_vec, t)
            t_vec_tot = np.append(t_vec_tot, t)
            dist_vecx= np.append(dist_vecx, distx)
            dist_vecy= np.append(dist_vecy, disty)
            dist_vecz= np.append(dist_vecz, distz)

            force = agent.forward(state)
            Agent_action = force2torques(env, force)
            ReL_action ,_, _, _, e0, e1= agent.ReL.action(env.robots, env._eef0_xmat, env._eef1_xmat, tg0, tg1, nd0, nd1)
            
            if t_pre<t<t_pre+250:
                csi += 0.004

            #print(csi)
            if t>=t_pre:
                csi = min(1/(np.cosh(80*max(np.abs(e0[0][0]), np.abs(e1[0][0])))), csi)
                tau = 1 + (1-csi)/20
                    
            obs, reward, done, _, = agent.env.step(tau*ReL_action + csi*Agent_action)
            #obs, reward, done, _, = agent.env.step(ReL_action + Agent_action)
            
            env.Rel_intervention = csi
            state= agent.get_state()
            env.render()
            episode_return += reward * math.pow(1.0, t)
            reward_vec = np.append(reward_vec, reward)
            reward_norm_dist_vec = np.append(reward_norm_dist_vec , env.reward_norm_dist)
            reward_velocity_vec =  np.append(reward_velocity_vec , env.reward_velocity)
            reward_terminal_state_vec =  np.append(reward_terminal_state_vec , env.reward_terminal_state)
            reward_increment_vec =  np.append(reward_increment_vec , env.reward_increment)
            reward_ReL_intervention_vec = np.append(reward_ReL_intervention_vec, env.ReL_intervention_reward)
            residual_filter0 = np.append(residual_filter0, e0)
            residual_filter1 = np.append(residual_filter1, e1)
            tangential_force0 = np.append(tangential_force0, tg0)
            tangential_force1 = np.append(tangential_force1, tg1)

    print("Episode Return: ", episode_return)
    dist_vec= np.delete(dist_vec, 0)
    t_vec = np.delete(t_vec, 0)
    dist_vecx= np.delete(dist_vecx, 0)
    dist_vecy= np.delete(dist_vecy, 0)
    dist_vecz= np.delete(dist_vecz, 0)
    reward_vec = np.delete(reward_vec, 0)
    reward_norm_dist_vec = np.delete(reward_norm_dist_vec , 0)
    reward_velocity_vec =  np.delete(reward_velocity_vec , 0)
    reward_terminal_state_vec =  np.delete(reward_terminal_state_vec , 0)
    reward_increment_vec =  np.delete(reward_increment_vec , 0)
    reward_ReL_intervention_vec = np.delete(reward_ReL_intervention_vec, 0)
    residual_filter0 = np.delete(residual_filter0, 0)
    residual_filter1 = np.delete(residual_filter1, 0)
    tangential_force0 = np.delete(tangential_force0, 0)
    tangential_force1 = np.delete(tangential_force1, 0)
    t_vec_tot = np.delete(t_vec_tot, 0)
    pos_vec = np.delete(pos_vec, 0)

    plt.figure(1, figsize=(10, 6))
    plt.plot(t_vec, dist_vecz, 'b', label='Z Distance', linewidth=2)
    plt.title('Distance vs. Iterations', fontsize=20) 
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Distance [m]', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--')
    plt.show()

    plt.figure(2, figsize=(10, 6))
    plt.plot(t_vec, pos_vec, 'b', label='Object positionn', linewidth=2)
    plt.axhline(env.target_pos[2]+(0.05*offs_pos[2]), color='k', linestyle='--', linewidth=1)
    plt.axhline(env.target_pos[2]-(0.05*offs_pos[2]), color='k', linestyle='--', linewidth=1)
    plt.axhline(env.target_pos[2], color='r', linestyle='--', linewidth=1)
    plt.title('Distance vs. Iterations', fontsize=20) 
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Distance [m]', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--')
    plt.show()

    plt.figure(3, figsize=(10, 6))
    plt.plot(t_vec, reward_vec, 'k', label='Total Reward', linewidth=2)
    plt.plot(t_vec, reward_norm_dist_vec, 'r--', label='Norm Distance Reward', linewidth=2)
    plt.plot(t_vec, reward_velocity_vec, 'g--', label='Velocity Reward', linewidth=2)
    plt.plot(t_vec, reward_terminal_state_vec, 'b--', label='Terminal State Reward', linewidth=2)
    plt.plot(t_vec, reward_ReL_intervention_vec, 'm--', label='ReL Reward', linewidth=2)
    plt.title('Episode Reward Contribution at each timestep', fontsize=20)
    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Episode Return [m]', fontsize=14)  
    plt.legend(loc='upper left') 
    plt.grid(True, linestyle='--')
    plt.show()

    plt.figure(4, figsize=(10, 6))
    plt.plot(t_vec_tot, residual_filter0, 'r', label='Residual 0', linewidth=2)
    plt.plot(t_vec_tot, residual_filter1, 'b', label='Residual 1', linewidth=2)
    plt.title('Residual', fontsize=20)
    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Residual [N]', fontsize=14)  
    plt.legend(loc='upper left') 
    plt.grid(True, linestyle='--')
    plt.show()

    plt.figure(5, figsize=(10, 6))
    plt.plot(t_vec_tot, tangential_force0, 'r', label='Measured Tangential Force 0', linewidth=2)
    plt.plot(t_vec_tot, tangential_force1, 'b', label='Measured Tangential Force 1', linewidth=2)
    plt.title('Measure Tangential Force', fontsize=20)
    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Tangential force [N]', fontsize=14)  
    plt.legend(loc='upper left') 
    plt.grid(True, linestyle='--')
    plt.show()