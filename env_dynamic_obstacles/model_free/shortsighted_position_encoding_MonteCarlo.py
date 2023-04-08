# Determinstic start, shortsighted agent, position encoding based state, model free Monte Carlo

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



### function to update q_table (Monte Carlo)
def update_q_table(q_table, visit_counts, episode_trace, episode_return, df):
    for posenc, action, reward in episode_trace:
        key = posenc + [action]
        key = str(key)
        # update visit count
        if key in visit_counts:
            visit_counts[key] += 1
        else:
            visit_counts[key] = 1
        # update q-value
        if key in q_table:
            q_table[key] = (((visit_counts[key] - 1) * q_table[key]) + episode_return) / visit_counts[key]
        else:
            q_table[key] = episode_return
        # update episode_return for next step
        episode_return = (episode_return - reward) / df
    return q_table



### runner function
def run(env, num_episodes, eps, df):

    # Keeps track of useful statistics
    episode_lengths=np.zeros(num_episodes)
    episode_rewards=np.zeros(num_episodes)

    # Q-Table for Monte Carlo
    q_table = {}
    # table for maintaining visit counts
    visit_counts = {}
    # dacey rate for epsilon
    eps_decay_epoch = num_episodes / 10

    for i_episode in range(num_episodes):
        # if i_episode % 1 == 0:
        #     print('\nEpisode: {}/{}'.format(i_episode, num_episodes))

        # episode trace to record episode - (state, action, reward) triplet
        episode_trace = []
        # variable to store (discounted) episode return
        episode_return = 0
        # variable to count episode step
        episode_step = 0

        # epsilon decay
        if i_episode % eps_decay_epoch == 0:
            eps -= 0.1

        # deterministic start state in the episode
        state = env.reset()

        # deterministic position encoding at the start
        # posenc[0] = position co-ordinates
        # posenc[1] = agent's orientation
        posenc = [[0,0], 0]

        # act according to current policy and collect episode experience
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

            # append experience to episode trace
            episode_trace.append([posenc, action, reward])
            # update episode return
            episode_return += (df**episode_step) * reward
            episode_step += 1

            state = next_state
            posenc = next_posenc
            episode_lengths[i_episode] += 1
            episode_rewards[i_episode] += reward
            if done:
                break

        # offline learning
        q_table = update_q_table(q_table, visit_counts, episode_trace, episode_return, df)


    env.close()

    return q_table, episode_lengths, episode_rewards



# main
env = gym.make('MiniGrid-Empty-8x8-v0')
num_episodes = 1000
eps = 1 # decayed
# lr = 0.01
df = 0.99
q_table, episode_lengths, episode_rewards = run(env, num_episodes, eps, df)

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
fig.savefig('./plots/deterministic_shortsighted_montecarlo.png')
