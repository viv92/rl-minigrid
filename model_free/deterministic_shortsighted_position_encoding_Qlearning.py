# Determinstic start, shortsighted agent, position encoding based state, model free Q-learning

import gym
import gym_minigrid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random


### function to get next positional encoding - using determinism and human prior knowledge
def get_next_posenc(posenc, state, action, next_state):
    next_orientation = next_state['direction']
    current_orientation = state['direction']
    curr_x = posenc[0][0]
    curr_y = posenc[0][1]

    if action == 2: # forward
        # check if wall in front
        object_in_front = state['image'][3][-2][0]
        if not (object_in_front == 2) : # no wall
            if current_orientation == 0:
                curr_x += 1
            elif current_orientation == 1:
                curr_y += 1
            elif current_orientation == 2:
                curr_x -= 1
            else:
                curr_y -= 1
    next_posenc = [[curr_x, curr_y], next_orientation]
    return next_posenc


### function to get best action - argmax over q values
def get_best_action(env, posenc, q_table):
    all_actions = np.arange(env.action_space.n)
    all_q_values = np.zeros(len(all_actions))
    for action in all_actions:
        key = posenc + [action]
        key = str(key)
        if key in q_table:
            all_q_values[action] = q_table[key]
    best_action = np.argmax(all_q_values)
    return best_action

### function to return epsilon greedy action
def get_epsgreedy_action(env, best_action, eps):
    action = best_action
    random_prob = random.uniform(0,1)
    if random_prob < eps:
        action = np.random.choice(env.action_space.n)
    return action



### function to update q_table (Qlearning)
def update_q_table(env, q_table, posenc, action, reward, next_posenc, lr, df):
    all_actions = np.arange(env.action_space.n)
    all_next_q_values = np.zeros(len(all_actions))
    # get best next q-value
    for next_action in all_actions:
        next_key = next_posenc + [next_action]
        next_key = str(next_key)
        if next_key in q_table:
            all_next_q_values[next_action] = q_table[next_key]
    best_next_q_value = np.max(all_next_q_values)
    # update q-table
    key = posenc + [action]
    key = str(key)
    if key not in q_table:
        q_table[key] = 0
    q_table[key] = q_table[key] + lr * (reward + (df * best_next_q_value) - q_table[key])
    return q_table



### runner function
def run(env, num_episodes, eps, lr, df):

    # Keeps track of useful statistics
    episode_lengths=np.zeros(num_episodes)
    episode_rewards=np.zeros(num_episodes)

    # Q-Table for Qlearning
    q_table = {}

    eps_decay_epoch = num_episodes / 10

    for i_episode in range(num_episodes):
        # if i_episode % 1 == 0:
        #     print('\nEpisode: {}/{}'.format(i_episode, num_episodes))

        # epsilon decay
        if i_episode % eps_decay_epoch == 0:
            eps -= 0.1

        # deterministic start state in the episode
        state = env.reset()

        # deterministic position encoding at the start
        # posenc[0] = position co-ordinates
        # posenc[1] = agent's orientation
        posenc = [[0,0], 0]

        while True:
            # view last 10 episodes
            # if num_episodes - i_episode < 10:
            #     env.render()

            # get greedy action
            best_action = get_best_action(env, posenc, q_table)
            # choose action to be taken in epsilon-greedy fashion
            action = get_epsgreedy_action(env, best_action, eps)
            # take action
            next_state, reward, done, _ = env.step(action)
            # get positional encoding for next state
            next_posenc = get_next_posenc(posenc, state, action, next_state)
            # update Q-Table (Q-learning)
            q_table = update_q_table(env, q_table, posenc, action, reward, next_posenc, lr, df)

            # print('next_posenc: ', next_posenc)
            # print('state_transition_model:\n', state_transition_model)
            # print('q_table:\n', q_table)

            state = next_state
            posenc = next_posenc
            episode_lengths[i_episode] += 1
            episode_rewards[i_episode] += reward
            if done:
                break

    env.close()

    return q_table, episode_lengths, episode_rewards



# main
env = gym.make('MiniGrid-Empty-8x8-v0')
num_episodes = 1000
eps = 1 # decayed
lr = 0.01
df = 0.99
q_table, episode_lengths, episode_rewards = run(env, num_episodes, eps, lr, df)

# print('q_table:\n', q_table)

# plot results
fig = plt.figure()
x = np.arange(len(episode_lengths))
plt.subplot(2,1,1)
plt.plot(x, episode_lengths, label='ep_length')
plt.legend()
plt.subplot(2,1,2)
plt.plot(x, episode_rewards, label='ep_undiscounted_return')
plt.legend()
fig.savefig('./plots/deterministic_shortsighted_qlearning.png')
