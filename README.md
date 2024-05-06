# Adaptive-Reinforcement-Learning-for-Local-Dynamic-Path-Planning-in-Autonomous-Driving
A Bachelor's thesis project at the German University in Cairo (GUC) where an adaptive TD3 Model was implemented and trained to tackle the task of dynamic local path planning in autonomous vehicles
# Getting started 
To use this project, follow these steps:
1. Clone the repository using the following command:
    ```bash
    git clone https://github.com/your-username/Adaptive-Reinforcement-Learning-for-Local-Dynamic-Path-Planning-in-Autonomous-Driving.git
   ```
2. Run the `TD3_PG_Drive.ipynb` file in sequential order to train and test the TD3 model designed to solve the local dynamic path planning.

# Introduction and motivation
For Avs to plot their path and self-pilot, a plethora of components and software must function together in harmony. Navigation systems like GPS are needed to plan a global route for the vehicle and determine its real-time position. Sensors, like Lidar, radar, and cameras, are needed to be able to detect the environment around the car. Perception algorithms, computer vision, and sensor fusion are used to help the agent get a better understanding of their surroundings, however, a challenge arises for the agent when planning the route locally in real-time settings with rapidly changing obstacles, as it needs to make split-second decisions in order to assure the passenger's comfort and safety.

# Aim 
In my research, it was focused on Reinforcement Learning for its ability to emulate human training techniques by adjusting through trial and error. This approach eliminates the requirement for extensive training data, as autonomous systems can be placed in a training environment and taught to drive using a predefined reward function. I designed and tweaked a TD3 model to tackle the problem of local path planning and to deal with the dynamic bit of the challenge, an adaptive module was integrated to allow the model to keep up with new situations.

In particular, we focused on a state-of-the-art Twin Delayed Deep Deterministic Policy Gradient (TD3) for its effectiveness in handling continuous action spaces. This capability allows the autonomous agent to make subtle adjustments, ensuring both safety and comfort during the driving task. Moreover, TD3 addresses the issue of function approximation commonly found in standard DDPG models by employing twin critic deep neural networks. This improvement enhances stability, thanks to the implementation of a delayed update mechanism.

# Model structure 
In the model, we have 6 neural networks. The actor-network selects actions based on the current state, while the critic networks estimate the expected cumulative reward for those actions. To ensure stability during training, we use target networks that are periodically updated and serve as replicas of the actor and critic networks. 

During training, the actor-network takes the current state and determines appropriate actions. The critic networks then evaluate the expected reward by considering the state-action pair. We calculate a critical loss and incorporate Gaussian noise to encourage exploration and avoid getting stuck in local optima. This loss is used to update the weights of both the actor and critic networks.

# Model Training 
During training, our autonomous driving agent used the MetaDrive simulator, which provided real-time navigation data, lidar, and camera inputs. The actor network processed the current state and selected actions, while the critic networks evaluated expected rewards for state-action pairs. Gaussian noise was added to encourage exploration and avoid local optima. Transitions, including states, actions, next states, and rewards, were stored in a global replay buffer for training. After each episode, the agent trained on random batches from the replay buffer to improve performance. Extensive training steps exposed the agent to a wide range of driving situations, ensuring its ability to handle diverse challenges.

# Adaptive Module 
In order to address the static nature of the model and enable adaptation to changing situations, we employ a hybrid approach that combines online and batch learning techniques. Our model is trained instantly if its performance falls below a predefined threshold, ensuring immediate adjustment to new challenges. Additionally, training is conducted after a certain batch size of episodes to capture environment trends. Moreover, we enhance the reward function by introducing a lateral reward multiplier, motivating the agent to maintain a central position within the lane instead of oscillating back and forth. This comprehensive approach empowers our model to dynamically adapt and perform optimally in evolving scenarios.

# Future Work 
While our research has shown promising results in dynamic path planning for autonomous vehicles using adaptive reinforcement learning, there are still some limitations and avenues for future work to consider:

- **Computation Time and Resources:** One of the main limitations of our approach is the significant computation time and resources required for training the model. The training process can be computationally expensive, hindering scalability and practical implementation. Future work could explore the possibility of creating a hybrid approach that combines our adaptive reinforcement learning with supervised reinforcement learning techniques. This hybrid approach may help reduce the required training time and computational resources.

- **Limited Top Speed:** During our testing, we imposed a top speed limitation of 40 km/h to facilitate faster learning for the model. However, in real-world scenarios, autonomous vehicles often operate at higher speeds. Therefore, future work should aim to increase the top speed that the model can reach during training and testing. This would ensure that the model can effectively plan paths at higher velocities, leading to more realistic and practical results.

- **Additional Environmental Factors:** Our research primarily focused on dynamic path planning without considering other environmental factors such as weather conditions and lighting. Including these factors in future work would provide a more comprehensive evaluation of the model's performance. By incorporating weather conditions like rain, fog, or snow, and accounting for different lighting conditions, the model's ability to adapt to various real-world scenarios can be further assessed and improved.

# Contributing
Contributions to this project are welcome. If you would like to contribute, please follow these guidelines:
1. Fork the repository and create your branch.
2. Make your changes and test them thoroughly.
3. Create a pull request explaining the changes you have made.

# [Bachelor Thesis](https://drive.google.com/file/d/16GHCec4T0BO8XM4cqjz1L9-c0ZB6gpOu/view?usp=sharing)
# [Research Paper : Dynamic Path Planning for Autonomous Vehicles Using Adaptive Reinforcement Learning](https://www.researchgate.net/publication/378826512_Dynamic_Path_Planning_for_Autonomous_Vehicles_Using_Adaptive_Reinforcement_Learning)
