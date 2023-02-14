# Reinforcement-Learning-algorithms

This repository contains implementations of various reinforcement learning algorithms. The algorithms include:
- Dynamic Programming

### Prerequisites
The code is written in Python and requires the following packages:
- Numpy
- Matplotlib
- csv
- time

### Usage
To run a specific algorithm, simply navigate to the relevant folder and run the main.py python file. For example, to run the Dynamic Programming implementations, navigate to the Dynamic Programming folder and run main.py in your terminal or command prompt. The map of the world is defined in map.csv file as a grid(1 represents obstacles while 0 represents free-spaces). You can alter the map according to your needs.

**Note:** The robot can move in all 8 directions from each cell

- In the deterministic case, the robot executes all movements perfectly.
- In stochastic case, the robot has a 20% probability of going +-45 degrees from the commanded movement.

### Reward Distribution
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/map_DP.png)

The black areas are obstacles hitting which, the agent get a reward of -50. The white square is the goal which has a reward of 100, and all other areas have a reward of -1.

### Results

#### Dynamic Programming

##### Deterministic case
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/Discrete_VI.png)


 


