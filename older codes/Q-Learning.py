import numpy as np
import copy 


rewards = np.array([[-float('inf'), -float('inf'), 0, 100, -float('inf'), -float('inf'), -float('inf')],
     [-float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), 0],
     [0, -float('inf'), -float('inf'), 1000, -float('inf'), 100, -float('inf')],
     [0, -float('inf'), 0, -float('inf'), 10, -float('inf'), -float('inf')],
     [-float('inf'), 0, -float('inf'), 0, -float('inf'), 100, 0],
     [-float('inf'), -float('inf'), 0, -float('inf'), 0, 100, 0],
     [-float('inf'), 0, -float('inf'), -float('inf'), 0, 100, -float('inf')],])

# Parameters
gamma = 0.8
alpha = 0.01
num_episode = 50000
min_difference = 1e-3
goal_state = 5


def QLearning(rewards, goal_state=None, gamma=0.99, alpha=0.01, num_episode=1000, min_difference=1e-5):
    """ 
    Run Q-learning loop for num_episode iterations or till difference between Q is below min_difference.
    """
    Q = np.zeros(rewards.shape)
    all_states = np.arange(len(rewards)) 
    for i in range(num_episode):
        Q_old = copy.deepcopy(Q)
        # initialize state
        initial_state = np.random.choice(all_states)
        action = np.random.choice(np.where(rewards[initial_state] != -float('inf'))[0])
        Q[initial_state][action] = Q[initial_state][action] + alpha * (rewards[initial_state][action] + gamma * np.max(Q[action]) - Q[initial_state][action])
        cur_state = action
        # loop for each step of episode, until reaching goal state
        while cur_state != goal_state:
            # choose action form states using policy derived from Q
            action = np.random.choice(np.where(rewards[cur_state] != -float('inf'))[0])
            Q[cur_state][action] = Q[cur_state][action] + alpha * (rewards[cur_state][action] + gamma * np.max(Q[action]) - Q[cur_state][action])
            cur_state = action
        # break the loop if converge
        diff = np.sum(Q - Q_old)
        if diff < min_difference:
            break
    return np.around(Q/np.max(Q)*100)


Q = QLearning(rewards, goal_state=goal_state, gamma=gamma, alpha=alpha, num_episode=num_episode, min_difference=min_difference)


print(Q)
