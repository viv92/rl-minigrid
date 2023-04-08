# Determinstic start, shortsighted agent, position encoding based table lookup model, simulation based planning with monte carlo tree search
# MCTS with sepatate tree policy and rollout policy, value backup after full rollout (till terminal state or rollout_maxsteps)

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


### function to update q_table for the current state from MCTS estimate
def update_q_table(q_table, tree_q_table, posenc, num_actions, lr):
    for action in range(num_actions):
        key = str(posenc + [action])
        if key in tree_q_table:
            if key in q_table:
                q_table[key] += lr * ( tree_q_table[key] - q_table[key] ) # todo: verify if we should do a moving mean update or just a direct assignment
            else:
                q_table[key] = lr * tree_q_table[key]

    return q_table


### fuction to backup estimates at the end of an MCTS rollout
def MCTS_backup(trajectory, trajectory_return, tree_posencs, tree_q_table, tree_visit_counts, df):
    tree_expanded = False
    for posenc, action, reward in trajectory:
        if posenc in tree_posencs: # node exists in tree
            key = str(posenc + [action])

            if key in tree_visit_counts:
                tree_visit_counts[key] += 1
                tree_q_table[key] += (1/tree_visit_counts[key]) * (trajectory_return - tree_q_table[key])
            else:
                tree_visit_counts[key] = 1
                tree_q_table[key] = trajectory_return

        else: # node does not exist in tree
            if not tree_expanded: # add node to tree
                key = str(posenc + [action])
                tree_visit_counts[key] = 1
                tree_q_table[key] = trajectory_return
                tree_posencs.append(posenc)
                tree_expanded = True
            else: # tree is already expanded - done with backups
                break
        # for next step in trajectory
        trajectory_return = (trajectory_return - reward) / df
    return tree_posencs, tree_q_table, tree_visit_counts




### MCTS
def MCTS(posenc, state_transition_model, reward_model, planning_iterations, rollout_maxsteps, num_actions, eps, df, q_table, i_episode):
    # keeps track of posencs in the MCTS tree
    tree_posencs = []
    # tree visit counts - used for updating q-value from rollout estimate
    tree_visit_counts = {}
    # q_table for tree build in this MCTS
    tree_q_table = {}

    for i in range(planning_iterations):
        # trajectory conatiner - [state, action, reward]
        trajectory = []
        trajectory_length = 0
        trajectory_return = 0 # discounted return of the trajectory
        rollout_steps = 0 # keeps count of rollout steps
        rollout_done = False

        while not rollout_done:

            if posenc in tree_posencs:
                # get eps_greedy action
                best_action = get_best_action(num_actions, posenc, tree_q_table)
                action = get_epsgreedy_action(num_actions, best_action, eps)

                # check if experience is available in our model for this action
                key = posenc + [action]
                key = str(key)
                if key not in reward_model: # experience unavailable for this action in our model
                    # backup for next planning iteration
                    tree_posencs, tree_q_table, tree_visit_counts = MCTS_backup(trajectory, trajectory_return, tree_posencs, tree_q_table, tree_visit_counts, df)
                    rollout_done = True
                    continue

            else: # posenc outside tree
                # get random action (rollout policy)
                available_keys = []
                for tmp_action in range(num_actions):
                    tmp_key = posenc + [tmp_action]
                    tmp_key = str(tmp_key)
                    if tmp_key in reward_model:
                        available_keys.append([tmp_key, tmp_action])
                if len(available_keys) == 0: # experience unavailable for any action from this state in our model
                    # backup for next planning iteration
                    tree_posencs, tree_q_table, tree_visit_counts = MCTS_backup(trajectory, trajectory_return, tree_posencs, tree_q_table, tree_visit_counts, df)
                    rollout_done = True
                    continue
                else:
                    # pick a random action
                    random_index = random.randint(0, len(available_keys)-1)
                    key, action = available_keys[random_index]
                    rollout_steps += 1

            # take the action in the model
            reward = reward_model[key]
            trajectory_return += (df ** trajectory_length) * reward
            next_posenc, done = state_transition_model[key]
            # append to trajectory
            trajectory.append([posenc, action, reward])
            trajectory_length += 1

            if (rollout_steps > rollout_maxsteps) or (trajectory_length > trajectory_maxsteps) or done: # backup for next planning iteration

                if not done: # bootstrap trajectory return from value of next posenc
                    next_posenc_q_values = np.zeros(num_actions)
                    for tmp_action in range(num_actions):
                        key = str(next_posenc + [tmp_action])
                        if key in q_table:
                            next_posenc_q_values[tmp_action] = q_table[key]
                    trajectory_return += (df ** trajectory_length) * np.max(next_posenc_q_values)

                tree_posencs, tree_q_table, tree_visit_counts = MCTS_backup(trajectory, trajectory_return, tree_posencs, tree_q_table, tree_visit_counts, df)
                rollout_done = True
                continue

            # for next step in rollout
            posenc = next_posenc
            rollout_done = done

    return tree_q_table



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

            # MCTS
            tree_q_table = MCTS(posenc, state_transition_model, reward_model, planning_iterations, rollout_maxsteps, num_actions, eps, df, q_table, i_episode)
            # update q-values for current state (obtained from MCTS)
            q_table = update_q_table(q_table, tree_q_table, posenc, num_actions, lr)

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
planning_iterations = 10
rollout_maxsteps = 1 # decides depth/length of rollout trajectory
trajectory_maxsteps = 100 # to avoid infinite loops in planning
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
fig.savefig('./plots/corrected_MCTS_treePolicy_rolloutMaxsteps='+str(rollout_maxsteps)+'_planIter='+str(planning_iterations)+'.png')
