import matplotlib.pyplot as plt
import csv
import numpy as np
import time
import random
import math

class environment:
    def __init__(self):
        self.grid = self.load_map('map.csv')
        self.rewards = self.create_rewards(self.grid)

        #Deterministic actions
        self.actions = [[0,1,1],[1,0,1],[0,-1,1],[-1,0,1]] #right,down,left,up

    def load_map(self,file_path):
        grid = []
        # Load from the file
        with open(file_path, 'r') as map_file:
            reader = csv.reader(map_file)
            for i, row in enumerate(reader):
                # load the map
                int_row = [int(col) for col in row]
                grid.append(int_row)   
        return np.array(grid)

    def create_rewards(self,grid):
        rewards = np.ones((np.shape(grid)[0],np.shape(grid)[1]))*-1
        rewards = rewards - 99*grid
        # plt.imshow(rewards)
        # plt.show()
        return rewards

class algorithms:
    def __init__(self,states,rewards,actions):
        self.alpha = 0.3
        self.num_episodes = 800
        self.num_iter = 20
        self.gamma = 0.9
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.size_row = states.shape[0]
        self.size_col = states.shape[1]
        self.Q_learning()
        self.SARSA()

    def action_make(self,current_state):
        action_space_indices=[]
        for i,action in enumerate(self.actions):
            next_state = (current_state[0]+action[0],current_state[1]+action[1])
            if next_state[0]>= 0 and next_state[0]<=self.size_row-1 and next_state[1]>= 0 and next_state[1]<=self.size_col-1:
                action_space_indices.append(i)
        return action_space_indices

    def Q_learning(self):
        policy = np.ones((self.states.shape[0],self.states.shape[1],len(self.actions)))/len(self.actions)

        eps_init = 0.1
        start_state = (3,0)

        sum_of_rewards=np.zeros(self.num_episodes)
        
        # start_time = time.time()
        for iter in range(self.num_iter):
            sum_of_rewards_per_iter=[]
            action_value_fn = np.zeros((self.states.shape[0],self.states.shape[1],len(self.actions)))

            eps = eps_init
            for episode in range(self.num_episodes):
                
                state = start_state
                done = False
                sum = 0

                while done == False:

                    action_space_indices = self.action_make(state)

                    temp = random.random()
                    if temp > eps:
                        action_index = action_space_indices[np.argmax(action_value_fn[state[0],state[1],action_space_indices])]
                    else:
                        action_index = np.random.choice(action_space_indices)

                    next_state = (state[0]+self.actions[action_index][0],state[1]+self.actions[action_index][1])
                    rew = self.rewards[next_state[0],next_state[1]]
                    sum += rew
                    action_value_fn[state[0],state[1],action_index] += self.alpha*(rew + self.gamma*np.max(action_value_fn[next_state[0],next_state[1],self.action_make(next_state)]) - action_value_fn[state[0],state[1],action_index])
                    
                    state = next_state

                    if state == (3,11):
                        done = True

                    if state[0] == 3 and state[1] >= 1 and state[1] < 11:
                        state = start_state
                # eps = eps_init*math.exp(-0.01*episode)
                sum_of_rewards_per_iter.append(sum)
            sum_of_rewards += np.array(sum_of_rewards_per_iter)/self.num_iter
            
        fig = plt.figure()
        fig.suptitle("Q-learning")
        plt.plot(list(range(1,self.num_episodes+1)),sum_of_rewards,label = "Q-learning")
        plt.legend()
        plt.xlabel("Number of episodes")
        plt.ylabel("Sum of rewards")
        plt.ylim([-100,0])
        plt.show()

        path = np.zeros((self.size_row,self.size_col))

        state = start_state
        path[state[0],state[1]] = 1
        done_path = False
        while done_path == False:
            action_space_indices = self.action_make(state)
            action_index = action_space_indices[np.argmax(action_value_fn[state[0],state[1],action_space_indices])]
            next_state = (state[0]+self.actions[action_index][0],state[1]+self.actions[action_index][1])            
            state = next_state
            if state == (3,11):
                done_path = True
            path[state[0],state[1]] = 1    
        plt.imshow(path)
        plt.show()


    def SARSA(self):
        policy = np.ones((self.states.shape[0],self.states.shape[1],len(self.actions)))/len(self.actions)

        eps_init = 0.1
        start_state = (3,0)

        sum_of_rewards=np.zeros(self.num_episodes)
        action_value_fn = np.zeros((self.states.shape[0],self.states.shape[1],len(self.actions)))
        
        # start_time = time.time()
        for iter in range(self.num_iter):
            sum_of_rewards_per_iter=[]
            action_value_fn = np.zeros((self.states.shape[0],self.states.shape[1],len(self.actions)))
            eps = eps_init
            for episode in range(self.num_episodes):
                state = start_state
                done = False
                sum = 0

                action_space_indices = self.action_make(state)
                temp = random.random()
                if temp > eps:
                    action_index = action_space_indices[np.argmax(action_value_fn[state[0],state[1],action_space_indices])]
                else:
                    action_index = np.random.choice(action_space_indices)

                while done == False:
                    next_state = (state[0]+self.actions[action_index][0],state[1]+self.actions[action_index][1])
                    rew = self.rewards[next_state[0],next_state[1]]
                    sum += rew

                    action_space_indices_next = self.action_make(next_state)

                    temp = random.random()
                    if temp > eps:
                        action_index_next = action_space_indices_next[np.argmax(action_value_fn[next_state[0],next_state[1],action_space_indices_next])]
                    else:
                        action_index_next = np.random.choice(action_space_indices_next)

                    action_value_fn[state[0],state[1],action_index] += self.alpha*(rew + self.gamma*(action_value_fn[next_state[0],next_state[1],action_index_next]) - action_value_fn[state[0],state[1],action_index])
                    
                    state = next_state
                    action_index = action_index_next

                    if state == (3,11):
                        done = True

                    if state[0] == 3 and state[1] >= 1 and state[1] < 11:
                        state = start_state
                sum_of_rewards_per_iter.append(sum)
                # eps = eps_init*math.exp(-0.1*episode)

            # print(len(sum_of_rewards_per_iter))
            sum_of_rewards += np.array(sum_of_rewards_per_iter)/self.num_iter
        # print(action_value_fn)     
        fig = plt.figure()
        fig.suptitle("SARSA")
        plt.plot(list(range(1,self.num_episodes+1)),sum_of_rewards,label = "SARSA")
        plt.legend()
        plt.xlabel("Number of episodes")
        plt.ylabel("Sum of rewards")
        plt.ylim([-100,0])
        plt.show()
            
        path = np.zeros((self.size_row,self.size_col))

        state = start_state
        path[state[0],state[1]] = 1
        done_path = False
        while done_path == False:
            action_space_indices = self.action_make(state)
            action_index = action_space_indices[np.argmax(action_value_fn[state[0],state[1],action_space_indices])]
            next_state = (state[0]+self.actions[action_index][0],state[1]+self.actions[action_index][1])            
            state = next_state
            if state == (3,11):
                done_path = True
            path[state[0],state[1]] = 1    
        plt.imshow(path)
        plt.show()
            
            
if __name__ == "__main__":
    env_obj = environment()
    alg_obj = algorithms(env_obj.grid,env_obj.rewards,env_obj.actions)
    