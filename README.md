# Reinforcement-Learning-algorithms

This repository contains implementations of various reinforcement learning algorithms. The algorithms include:
- Dynamic Programming
- Monte Carlo
- SARSA, Q-Learning
- DynaQ

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
Time taken for Value Iteration to converge is 0.4560210704803467 seconds.

![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/Discrete_PI.png)
Time taken for Policy Iteration to converge is 1.5746524333953857 seconds.


![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/Discrete_GPI.png)
Time taken for Generalized Policy Iteration to converge is 0.27486228942871094 seconds.

##### Stochastic case
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/Sto_VI.png)
Time taken for Value Iteration to converge is 1.2053732872009277 seconds.

![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/Sto_PI.png)
Time taken for Policy Iteration to converge is 4.194683790206909 seconds.

![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/Sto_GPI.png)
Time taken for Generalized Policy Iteration to converge is 0.9754695892333984 seconds.

#### Monte Carlo
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/MCES.png)
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/MC_eps.png)

#### Q-Learning and SARSA
Here we see the results of these two algorithms on the cliff-walking problem
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/cliff_walking1.png)
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/plots.png)

#### Dyna-Q
![alt text](https://github.com/mayankbansal82/Reinforcement-Learning-algorithms/blob/main/images/dynaq.png)
As the number of steps are increased the number of episodes required to obtain optimal policy is decreased.





 


