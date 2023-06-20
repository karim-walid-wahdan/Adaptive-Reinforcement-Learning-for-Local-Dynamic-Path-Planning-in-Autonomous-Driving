import numpy as np
import copy
from typing import Protocol


class Graph(Protocol):
    def neighbors(self, id: float) -> list[float]: pass


class WeightedGraph(Graph):
    def cost(self, from_id: float, to_id: float) -> float: return 1


def QLearning(rewards ,start =None, goal=None, gamma=0.99, alpha=0.01, num_episode=1000, min_difference=1e-5):
    """
    Run Q-learning loop for num_episode iterations or till difference between Q is below min_difference.
    """
    Q = np.zeros(rewards.shape)
    all_states = np.arange(len(rewards))

    for i in range(num_episode):
        Q_old = copy.deepcopy(Q)
        # initialize state
        action = np.random.choice(np.where(rewards[start] != -float('inf'))[0])
        Q[start][action] = Q[start][action] + alpha * (rewards[start][action] + gamma * np.max(Q[action]) - Q[start][action])
        cur_state = action
        # loop for each step of episode, until reaching goal state
        parent = cur_state
        while cur_state != goal:
            # choose action form states using policy derived from Q
            action = np.random.choice(np.where(rewards[cur_state] != -float('inf'))[0])
            while action == parent:
                action = np.random.choice(np.where(rewards[cur_state] != -float('inf'))[0])
            Q[cur_state][action] = Q[cur_state][action] + alpha * (
                    rewards[cur_state][action] + gamma * np.max(Q[action]) - Q[cur_state][action])
            cur_state = action
            parent = cur_state
        # break the loop if converge
        diff = np.sum(Q - Q_old)
        if diff < min_difference:
            break
    return np.around(Q / np.max(Q) * 100)


# creating a demo road network of 9 nodes
"""
    0<--->1<--->2
    ↕     ↕     ↕ 
    3<--->4<--->5
    ↕     ↕     ↕
    6<--->7<--->8
"""
road_network = WeightedGraph();
road_network.neighbors = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7],
}
# defining the cost of each neighbor to neighbor hob
road_network.cost = {
    (0, 1): 1, (0, 3): 1,
    (1, 0): 1, (1, 2): 5, (1, 4): 0.5,
    (2, 1): 1, (2, 5): 1,
    (3, 0): 1, (3, 4): 1, (3, 6): 1,
    (4, 1): 0.5, (4, 3): 4, (4, 5): 1, (4, 7): 1,
    (5, 2): 1, (5, 4): 1, (5, 8): 1,
    (6, 3): 1, (6, 7): 1,
    (7, 4): 1, (7, 6): 1, (7, 8): 1,
    (8, 5): 1, (8, 7): 1,
}


# initializing a rewards array of size road_network.neighbors * road_network.neighbors with values negative infinity
rewards = np.full((len(road_network.neighbors), len(road_network.neighbors)), -float("inf"))

# Parameters
gamma = 0.8  #learning rate
alpha = 0.01  #discount factor
num_episode = 5000  # number of training episodes
min_difference = 1e-3
start = 1  #start node
goal = 3  #end node
for key, weight in road_network.cost.items():
    rewards[key] = 1 / weight
Q = QLearning(rewards, start=start,goal=goal, gamma=gamma, alpha=alpha, num_episode=num_episode,min_difference=min_difference)
print(Q)