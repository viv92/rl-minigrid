# dynamic obstacle environment

import gym
import gym_minigrid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(0)
np.random.seed(0)

env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
nA = env.action_space.n

num_episodes = 2

for ep in range(num_episodes):
    obs = env.reset()
    print('obs: ', obs)
    while True:
        env.render()
        action = np.random.randint(0, nA)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs

        # wait for keypress
        input()

        if done:
            break
