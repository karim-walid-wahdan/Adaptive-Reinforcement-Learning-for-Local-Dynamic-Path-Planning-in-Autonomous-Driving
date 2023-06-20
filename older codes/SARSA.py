import numpy as np
import gym

# FrozenLake-v0 gym environment
env = gym.make('FrozenLake-v1')

# Parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.05
gamma = 0.95

# Initializing the Q-vaue
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Function to choose the next action with episolon greedy


def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


# Initializing the reward
reward = 0

# Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)

    while t < max_steps:
        # Visualizing the training
        # env.render()

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        # Choosing the next action
        action2 = choose_action(state2)

        # Learning the Q-value
        Q[state1, action1] = Q[state1, action1] + alpha * \
            (reward + gamma * Q[state2, action2] - Q[state1, action1])

        state1 = state2
        action1 = action2

        # Updating the respective vaLues
        t += 1
        reward += 1

        # If at the end of learning process
        if done:
            break

# Evaluating the performance
print("Performace : ", reward/total_episodes)

# Visualizing the Q-matrix
print(Q)
