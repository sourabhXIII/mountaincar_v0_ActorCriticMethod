"""
@author sourabhxiii
"""

import gym
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from mcv0_utility.policymodel import PolicyModel
from mcv0_utility.valuemodel import ValueModel
from mcv0_utility.featuretransformer import FeatureTransformer
import matplotlib.pyplot as plt

# discount factor
GAMMA = 0.99

def play_game(env, pmodel, vmodel, e):
    # 1. pmodel takes axn
    # 2. collect reward (r)
    # 3. vmodel predicts value (V) of next state
    # 4. compute advantage score A = Q - V
        # Q = r + gamma*V
        # V = prediction of vmodel for previous state
    # 5. pmodel gets a learning (previous state, axn, A)
    # 6. vmodel gets a learning (previous state, Q)

    # 0
    env.seed(1)
    obs = env.reset()
    # print(env.action_space.sample())
    totalreward = 0
    loop = 0
    done = False
    axns_taken = np.zeros((3,1))
    max_timestep = 1000

    while done is False and loop < max_timestep:
        # env.render()
        # 1
        axn = pmodel.get_action(obs)
        # print(axn)
        axns_taken[axn] += 1
        # 2
        prev_obs = obs
        obs, r, done, _ = env.step(axn)
        totalreward += r
        # 3
        V = vmodel.predict(obs)
        # 4
        Q = r + GAMMA * V
        A = Q - vmodel.predict(prev_obs)
        # 5
        pmodel.update_weight(prev_obs, axn, A, ((e*max_timestep) + loop))
        vmodel.update_weight(prev_obs, Q)

        # inf
        loop += 1
    print('\t Actions taken:\n\t %s' % axns_taken)
    return totalreward, loop


def main():
    # 0. Create environment
        # know your environment
    # 1. create a FT object to transform the env to feature vector
    # 2. create policy model
    # 3. create value model
    # 4. play game
    
    # 0
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')

    # env.seed(1)
    env.reset()
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    # print(env.action_space.sample())
    # obs, r, done, _ = env.step(env.action_space.sample())

    # 1
    FT = FeatureTransformer(env, n_components=100)
    input_dim = FT.dimensions
    # 2
    pmodel = PolicyModel(input_dim, env.action_space.n ,FT)
    # pmodel = PolicyModel(input_dim, env.action_space.shape[0] ,FT)
    # 3
    vmodel = ValueModel(input_dim, FT)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    pmodel.set_session(sess)
    vmodel.set_session(sess)
    # 4
    episodes = 500
    rewards = np.empty(episodes)
    episode_lengths = np.empty(episodes)
    for e in range(episodes):
        episode_reward, steps = play_game(env, pmodel, vmodel, e)
        rewards[e] = episode_reward
        episode_lengths[e] = steps
        print('Episode %d reward collected %d' %(e, episode_reward))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')

    plt.subplot(212)
    plt.plot(episode_lengths)
    plt.ylabel('Length')
    plt.xlabel('Episodes')

    plt.show()

    # done with the game. close the environment
    env.close()
if __name__ == '__main__':
    main()


