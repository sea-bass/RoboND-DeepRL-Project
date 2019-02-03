
# Udacity Robotics Nanodegree: Deep RL Arm Manipulation
### Sebastian Castro, 2019

## Introduction
In this project, a deep reinforcement learning (RL) agent is trained to move a simulated robot manipulator toward a target object.

* The **observation** is an image showing a 2D projection of the robot arm environment
* The **output** is the joint position or velocity of the robot arm joints

To solve this problem, a [Deep-Q Network (DQN)](https://deepmind.com/research/dqn/) agent is used. This means that the output of the agent is one of several possible discrete actions. Depending on the mode used (position or velocity control), the DQN agent will increase or decrease the position or velocity of an individual joint at each time step.

## Setup
### Network Architecture
A provided convolutional neural network (CNN) architecture is used, which is provided in the project. This consists of 3 2-D convolutional layers with kernel size of 5 and stride of 2, with batch normalization and ReLU activations. This architecture can be found in the [`DQN.py`](https://github.com/sea-bass/RoboND-DeepRL-Project/blob/master/python/DQN.py) script.

### Reward Function
The reward function consists of 3 components:

* Losing reward (if hitting the ground or timing out): -50
* Winning reward (if object is reached): +50
* Interim reward: `10*avgGoalDist - 1`

where `avgGoalDist = alpha*goalDist - (1-alpha)*avgGoalDist`
and `alpha` = 0.5

NOTE: For the case where only the gripper base should touch the object, a winning reward of +50 is received if the gripper base collides with the object, and a "neutral" reward of 0 is received if another link of the arm collides with the object. This is done because it is a preferable outcome to timing out or hitting the ground.

### Hyperparameters
Training is done with the Adam optimizer, with an initial learning rate of 0.001. The input image is scaled to 256-by-256 pixels. Replay memory consists of a buffer of 5000 experiences and training is done in mini-batches of 16 experiences. The discount factor is 0.95, meaning the estimated value is halved in approximately 13.5 steps.

The agent explores using an epsilon-greedy approach, meaning it selects a random action with an initial probability of 0.9, which decays to a minimum of 0.05 after 200 training episodes.

Finally, we have chosen to use velocity control as the output. At each time step, the velocity of one joint can be increased or decreased by 0.05 radians per time step. The joint limits of the arm are also obeyed.

## Results
### Experiment 1: Any Part of Arm
Almost 100% accuracy!

Refer to [this video](https://github.com/sea-bass/RoboND-DeepRL-Project/raw/master/videos/RoboND-DeepRL-AnyLink.mp4").

### Experiment 2: Gripper Base Only
~90% accuracy. We created a #define

Refer to [this video](https://github.com/sea-bass/RoboND-DeepRL-Project/raw/master/videos/RoboND-DeepRL-GripperBar.mp4").

### Experiment 3: Random Prop Location
We attempted randomizing the initial object location, which required more training. To do this, we created a #define 

Refer to [this video](https://github.com/sea-bass/RoboND-DeepRL-Project/raw/master/videos/RoboND-DeepRL-RandomProp.mp4").

## Future Work / Conclusion
Randomizing the initial object location and moving to 3DOF, will require more training steps, more data, and maybe higher image resolution. Was running out of memory on GPU.

Changing default neural network structure.

Using continuous outputs with agents like DDPG, PPO, etc.
