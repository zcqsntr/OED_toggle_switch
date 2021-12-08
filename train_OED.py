import sys
import os
IMPORT_PATH = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RL_OED'), 'imports')
print(IMPORT_PATH)
sys.path.append(IMPORT_PATH)

import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from PG_agent import *
from DQN_agent import *
from xdot import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
import multiprocessing
import json
from scipy.stats import truncnorm

if __name__ == '__main__':
    save_path = './output'
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    n_cores = multiprocessing.cpu_count()
    print('Num CPU cores:', n_cores)

    params = json.load(open('./params.json'))

    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_sampling_intervals, sampling_time, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    print('input_bounds', np.array(input_bounds))

    actual_params = DM(actual_params)

    normaliser = np.array(normaliser)

    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements

    param_guesses = actual_params

    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0],
                       hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs,
                       hidden_layer_size[0], hidden_layer_size[1], 1]

    agent = DDPG_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes, policy_act=tf.nn.sigmoid,
                       val_learning_rate=0.0001, pol_learning_rate=pol_learning_rate)  # , pol_learning_rate=0.0001)


    total_time = N_sampling_intervals*sampling_time # minutes
    N_control_intervals = total_time//control_interval_time

    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = 11
    agent.mem_size = 500000000

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)

    print('ENV INTITIALISED')

    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)

    n_unstables = []
    all_returns = []
    all_test_returns = []
    agent.std = 0.1
    agent.noise_bounds = [-0.25, 0.25]
    agent.action_bounds = [0, 1]
    policy_delay = 2
    update_count = 0
    explore_rate = 1
    max_std = 1  # for exploring
    unstable = 0

    for episode in range(int(n_episodes // skip)):

        upper_bounds = np.array([0.4488,1.1220,0.3366,112.2018,336.6055,112.2018,5.1250,5.1250,1.7783,11.2202,336.6055,1.1220,5.1250,5.1250])
        lower_bounds = np.array([0.0036,0.0089,0.0027,0.8913,2.6738,0.8913,0,0,0.0056,0.0891,2.6738,0.0089,0,0])
        means = np.array([0.23, 0.57, 0.17, 56.55, 169.64, 56.55, 2.56, 2.56, 0.89, 5.65, 169.64, 0.57, 2.56, 2.56])
        stds = np.array([0.11, 0.28, 0.08, 27.83, 83.48, 27.83, 1.28, 1.28, 0.44, 2.78, 83.48, 0.28, 1.28, 1.28])
        actual_params = np.random.normal(means, stds, size=(skip, len(means)))

        # clip params
        #while not np.all( lower_bounds < param_guesses <upper_bounds):
        #   actual_params = np.random.normal(means, stds)

        env.param_guesses = DM(actual_params)

        states = [env.get_initial_RL_state_parallel() for i in range(skip)]

        e_returns = [0 for _ in range(skip)]

        e_actions = []

        e_exploit_flags = []
        e_rewards = [[] for _ in range(skip)]
        e_us = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]

        sequences = [[[0] * pol_layer_sizes[1]] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]

        for e in range(0, N_sampling_intervals):

            ''' Get next experimental input and simualte the result'''
            inputs = [states, sequences]

            if e%(N_sampling_intervals//N_control_intervals) == 0: # if we need to get a new control
                actions = agent.get_actions0(inputs, explore_rate=explore_rate, test_episode=False, recurrent=True)

            e_actions.append(actions)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous=True)
            next_states = []

            '''Unpack the outputs of the parallel environments'''
            for i, o in enumerate(outputs):
                next_state, reward, done, _, u = o
                e_us[i].append(u)
                next_states.append(next_state)
                state = states[i]

                action = actions[i]

                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    # next_state = [0]*pol_layer_sizes[0] # maybe dont need this
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1:  # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward

            # print('sequences', np.array(sequences).shape)
            # print('sequences', sequences[0])
            states = next_states

            '''Check for instability'''
            for trajectory in trajectories:
                if np.all([np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))]) and not math.isnan(
                        np.sum(trajectory[-1][0])):  # check for instability
                    agent.memory.append(trajectory)  # monte carlo, fitted
                else:
                    unstable += 1
                    print('UNSTABLE!!!')
                    print((trajectory[-1][0]))

            '''Print output'''
            if episode > 1000 // skip:
                print('training', update_count)

                for hello in range(skip):
                    # print(e, episode, hello, update_count)
                    update_count += 1
                    policy = update_count % policy_delay == 0

                    agent.Q_update(policy=policy, fitted=fitted, recurrent=recurrent)
                print('fitting time', time.time() - t)

            explore_rate = DQN_agent.get_rate(None, episode, 0, 1, n_episodes / (11 * skip)) * max_std


    np.save(save_path + 'all_returns.npy', np.array(all_returns))

    np.save(save_path + 'n_unstables.npy', np.array(n_unstables))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    agent.save_network(save_path)