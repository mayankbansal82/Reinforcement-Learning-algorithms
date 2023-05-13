import matplotlib.pyplot as plt
import csv
import numpy as np
import time
import random
import math

class DynaQ:
    def __init__(self):
        self.grid = self.load_map('map.csv')
        # print("oub")
        self.rewards = self.create_rewards(self.grid)

        #Deterministic actions
        self.actions = [[-1,0,1],[1,0,1],[0,1,1],[0,-1,1]] #u d r l

        self.action_value_fn = np.zeros((self.grid.shape[0],self.grid.shape[1],len(self.actions)))

        self.model = np.zeros((self.grid.shape[0],self.grid.shape[1],len(self.actions),2))
        self.size_row = self.grid.shape[0]
        self.size_col = self.grid.shape[1]
        self.visited = np.zeros((self.grid.shape[0],self.grid.shape[1],len(self.actions)))
        self.alpha = 0.1
        self.gamma = 0.95
        self.num_episodes = 50

        steps0 = self.DynaQ(0)
        steps5 = self.DynaQ(5)
        steps50 = self.DynaQ(50)

        fig = plt.figure()
        fig.suptitle("Dyna-Q")
        plt.plot(list(range(1,self.num_episodes+1)),steps0,label = "0 planning steps")
        plt.plot(list(range(1,self.num_episodes+1)),steps5,label = "5 planning steps")
        plt.plot(list(range(1,self.num_episodes+1)),steps50,label = "50 planning steps")
        plt.legend()
        plt.ylabel("steps per episode")
        plt.xlabel("Episodes")
        # plt.ylim([0,2000])
        plt.show()

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
        rewards = np.zeros((np.shape(grid)[0],np.shape(grid)[1]))
        rewards[0][8]=1
        # plt.imshow(rewards)
        # plt.show()
        return rewards
    
    def action_make(self,current_state):
        action_space_indices=[]
        for i,action in enumerate(self.actions):
            next_state = (current_state[0]+action[0],current_state[1]+action[1])
            if next_state[0]>= 0 and next_state[0]<=self.size_row-1 and next_state[1]>= 0 and next_state[1]<=self.size_col-1:
                action_space_indices.append(i)
        return action_space_indices
    
    def DynaQ(self,n):
        start_state = (2,0)
        eps = 0.1
        steps = []
        prev_state = []
        # sum_of_rewards=np.zeros(self.num_episodes)
        it = 0
        for episode in range(self.num_episodes):
            # print(episode)

            state = start_state
            done = False
            
            step = 0

            while done == False:
                step += 1

                action_space_indices = self.action_make(state)

                temp = random.random()
                if temp > eps:
                    action_index = action_space_indices[np.argmax(self.action_value_fn[state[0],state[1],action_space_indices])]
                else:
                    action_index = np.random.choice(action_space_indices)

                next_state = (state[0]+self.actions[action_index][0],state[1]+self.actions[action_index][1])
                rew = self.rewards[next_state[0],next_state[1]]
                # sum += rew
                self.action_value_fn[state[0],state[1],action_index] += self.alpha*(rew + self.gamma*np.max(self.action_value_fn[next_state[0],next_state[1],self.action_make(next_state)]) - self.action_value_fn[state[0],state[1],action_index])
                
                self.model[state[0]][state[1]][action_index][0] = next_state[0]
                self.model[state[0]][state[1]][action_index][1] = next_state[1]

                # if(self.visited[state[0]][state[1]][action_index])==0:
                #     it+=1
                #     # print(it)
                #     self.visited[state[0]][state[1]][action_index]=1
                prev_state.append((state,action_index))


                for i in range(n):
                    idx = random.randrange(len(prev_state))
                    s = prev_state[idx][0]
                    a = prev_state[idx][1]
                    s_prime = (int(self.model[s[0],s[1],a,0]),int(self.model[s[0],s[1],a,1]))
                    # print(s_prime)
                    r = self.rewards[s_prime[0]][s_prime[1]]

                    self.action_value_fn[s[0],s[1],a] += self.alpha*(r + self.gamma*np.max(self.action_value_fn[s_prime[0],s_prime[1],self.action_make(s_prime)]) - self.action_value_fn[s[0],s[1],a])



                state = next_state

                if state == (0,8):
                    done = True
            steps.append(step)
        return steps
        # print(self.model)
        
            
if __name__ == "__main__":
    env_obj = DynaQ()

    