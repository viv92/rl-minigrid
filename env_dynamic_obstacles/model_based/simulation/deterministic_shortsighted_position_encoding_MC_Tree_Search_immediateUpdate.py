# Determinstic start, shortsighted agent, position encoding based table lookup model, simulation based planning with monte carlo tree search
# Simulation policy updated at EACH planning iteration

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



### function to update / learn table lookup models
def update_models(posenc, action, reward, next_posenc, done, state_transition_model, reward_model):
    # create dictionary key
    key = posenc + [action]
    key = str(key)
    if key in state_transition_model:
        # state-action pair already encountered - exit since deterministic environment
        return state_transition_model, reward_model
    # update model
    state_transition_model[key] = [next_posenc, done]
    reward_model[key] = reward
    return state_transition_model, reward_model



### function to get best action - argmax over q values
def get_best_action(num_actions, posenc, q_table):
    all_actions = np.arange(num_actions)
    all_q_values = np.zeros(len(all_actions))
    for action in all_actions:
        key = posenc + [action]
        key = str(key)
        if key in q_table:
            all_q_values[action] = q_table[key]
    best_action = np.argmax(all_q_values)
    return best_action

### function to return epsilon greedy action
def get_epsgreedy_action(num_actions, best_action, eps):
    action = best_action
    random_prob = random.uniform(0,1)
    if random_prob < eps:
        action = np.random.choice(num_actions)
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



### function to simulate a trajectory over learnt model - starting from the current state - using current simulation policy
def simulate_trajectory(posenc, state_transition_model, reward_model, q_table, num_actions, eps, df):
    # trajectory conatiner - [state, action, reward]
    trajectory = []
    trajectory_length = 0
    trajectory_return = 0 # discounted return of the trajectory
    done = False
    while not done:
        # get best action
        best_action = get_best_action(num_actions, posenc, q_table)
        # get epsilon greedy action
        ######  THIS ACTION CAN BE VIEWED AS OBTAINED FROM THE TREE POLICY #####
        action = get_epsgreedy_action(num_actions, best_action, eps)
        # check if experience is available in our model for this action
        key = posenc + [action]
        key = str(key)
        if key not in reward_model: # pick an action for which experience is available
            available_keys = []
            for tmp_action in range(num_actions):
                tmp_key = posenc + [tmp_action]
                tmp_key = str(tmp_key)
                if tmp_key in reward_model:
                    available_keys.append([tmp_key, tmp_action])
            if len(available_keys) == 0: # experience (model) not available for any action - cannot simulate further
                return trajectory, trajectory_return

            random_index = random.randint(0, len(available_keys)-1)
            ###### THIS ACTION CAN BE VIEWED AS OBTAINED FROM THE DEFAULT POLICY #####
            key, action = available_keys[random_index]
        # take the action in the model
        reward = reward_model[key]
        trajectory_return += (df ** trajectory_length) * reward
        next_posenc, done = state_transition_model[key]
        # append to trajectory
        trajectory.append([posenc, action, reward])
        trajectory_length += 1
        if trajectory_length > planning_trajectory_maxsteps:
            break
        # for next step
        posenc = next_posenc
    return trajectory, trajectory_return



### runner function
def run(env, num_episodes, planning_iterations, eps, lr, df):

    # Keeps track of useful statistics
    episode_lengths=np.zeros(num_episodes)
    episode_rewards=np.zeros(num_episodes)

    # table lookup models
    state_transition_model = {}
    reward_model = {}
    # Q-Table
    q_table = {}
    # table for maintaining visit counts
    visit_counts = {}
    # epsilon decay rate
    eps_decay_epoch = num_episodes / 10
    # number of actions in the discrete action space
    num_actions = env.action_space.n

    for i_episode in range(num_episodes):
        if i_episode % 1 == 0:
            print('\nEpisode: {}/{}'.format(i_episode, num_episodes))

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
            # # view last 10 episodes
            # if num_episodes - i_episode < 10:
            #     env.render()

            # simulation based planning - monte carlo tree search (using immediately updated simulation policy)
            for k in range(planning_iterations):
                # simulate a trajectory from current state - using current simulation policy
                trajectory, trajectory_return = simulate_trajectory(posenc, state_transition_model, reward_model, q_table, num_actions, eps, df)
                # immediately update q-table for all states in the trajectory - also updates the simulation policy at each planning iteration
                q_table = update_q_table(q_table, visit_counts, trajectory, trajectory_return, df)

            # get greedy action
            best_action = get_best_action(num_actions, posenc, q_table)
            # choose action to be taken in epsilon-greedy fashion
            action = get_epsgreedy_action(num_actions, best_action, eps)
            # take action
            next_state, reward, done, _ = env.step(action)
            # get positional encoding for next state
            next_posenc = get_next_posenc(posenc, state, action, next_state)

            # update table lookup models
            state_transition_model, reward_model = update_models(posenc, action, reward, next_posenc, done, state_transition_model, reward_model)

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
num_episodes = 300
planning_iterations = 2
planning_trajectory_maxsteps = 100 # to avoid infinitely long trajectory during planning
eps = 1 # decayed
lr = 0.01
df = 0.99
q_table, episode_lengths, episode_rewards = run(env, num_episodes, planning_iterations, eps, lr, df)

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
fig.savefig('./plots/deterministic_shortsighted_mc_tree_search_immediateUpdate.png')
