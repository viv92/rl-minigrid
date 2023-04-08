# Random agent

import gym
import gym_minigrid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

from collections import defaultdict

# matplotlib.style.use('ggplot')


def run(env, num_episodes):

    # Keeps track of useful statistics
    episode_lengths=np.zeros(num_episodes)
    episode_rewards=np.zeros(num_episodes)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1 == 0:
            print('\nEpisode: {}/{}'.format(i_episode, num_episodes))

        # start state in the episode
        state = env.reset()

        while True:
            # env.render()

            # choose random action
            action = np.random.choice(np.arange(env.action_space.n))

            # take action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_lengths[i_episode] += 1
            episode_rewards[i_episode] += reward
            if done:
                break

    env.close()

    return episode_lengths, episode_rewards



# main
env = gym.make('MiniGrid-Empty-8x8-v0')
episode_lengths, episode_rewards = run(env, 10)

# plot
x = np.arange(len(episode_lengths))
plt.plot(x, episode_lengths, label='ep_length')
plt.plot(x, episode_rewards, label='ep_return')
plt.legend()
plt.show()
