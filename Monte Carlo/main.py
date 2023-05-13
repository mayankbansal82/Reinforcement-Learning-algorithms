import matplotlib.pyplot as plt
import csv
import numpy as np
import time
import random
import math

class algorithms:
    def __init__(self):
        self.states = [0,1,2,3,4,5]
        self.actions = [-1,1]
        self.rewards = [1,0,0,0,0,5]
        # self.policy = np.ones((len(self.states),len(self.actions)))/len(self.actions)
        # self.action_value_fn = np.zeros((len(self.states),len(self.actions)))
        # self.count = np.zeros((len(self.states),len(self.actions)))
        self.num_episodes = 2000
        self.gamma = 0.99
        self.MCES()
        self.MC_eps()

    def MCES(self):
        policy = np.ones((len(self.states),len(self.actions)))/len(self.actions)
        action_value_fn = np.zeros((len(self.states),len(self.actions)))
        count = np.zeros((len(self.states),len(self.actions)))
        state_value_0=[]
        state_value_1=[]
        state_value_2=[]
        state_value_3=[]
        state_value_4=[]
        state_value_5=[]
        start_time = time.time()
        for episode in range(self.num_episodes):
            
            state = random.randint(0,4)+1#random.choice(self.states)

            action_index = random.randint(0,len(self.actions)-1)

            G = 0
            done = False
            states=[]
            actions=[]
            rewards=[]
            
            if state == 0 or state == 5:
                done = True

            while not done:
                states.append(state)
                actions.append(action_index)
                prob = np.random.random()
                # print(prob)
                if prob <= 0.8:
                    next_state = self.states[state+self.actions[action_index]]
                elif prob >0.8 and prob <=0.95:
                    next_state = state
                elif prob >0.95:
                    next_state = self.states[state-self.actions[action_index]] #Stochastic

                # next_state = self.states[state+self.actions[action_index]] #Deterministic

                rew = self.rewards[next_state]
                rewards.append(rew)

                state = next_state
                action_index = np.random.choice(range(len(self.actions)),size=1,p=policy[state,:])[0]
    
                if next_state == 5 or next_state == 0:
                    done = True
            # print("states",states)
            # print("actions",actions)
            # print("rewards",rewards)

            G = 0
            T = len(states)
            
            for t in reversed(range(T)):
                G = self.gamma*G + rewards[t]
                found = False
                for i in range(t):
                    if states[i] == states[t] and actions[i] == actions[t]:
                        found = True
                if not found:
                    count[states[t]][actions[t]] += 1
                    action_value_fn[states[t]][actions[t]] += (G - action_value_fn[states[t]][actions[t]])/count[states[t]][actions[t]]
                    policy[states[t],:] = 0
                    policy[states[t],np.argmax(action_value_fn[states[t],:])] = 1
            state_value_0.append(policy[0][0]*action_value_fn[0][0] + policy[0][1]*action_value_fn[0][1])
            state_value_1.append(policy[1][0]*action_value_fn[1][0] + policy[1][1]*action_value_fn[1][1])
            state_value_2.append(policy[2][0]*action_value_fn[2][0] + policy[2][1]*action_value_fn[2][1])
            state_value_3.append(policy[3][0]*action_value_fn[3][0] + policy[3][1]*action_value_fn[3][1])
            state_value_4.append(policy[4][0]*action_value_fn[4][0] + policy[4][1]*action_value_fn[4][1])
            state_value_5.append(policy[5][0]*action_value_fn[5][0] + policy[5][1]*action_value_fn[5][1])
        total_time = time.time() - start_time
        print('Time taken for Monte Carlo Exploring Starts to converge is ',total_time)

        print(policy)
        print(action_value_fn)

        fig = plt.figure()
        fig.suptitle("Exploring Starts Stochastic")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_0,label = "State 0")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_1,label = "State 1")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_2,label = "State 2")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_3,label = "State 3")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_4,label = "State 4")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_5,label = "State 5")
        plt.legend()
        plt.xlabel("Number of episodes")
        plt.ylabel("State values")
        plt.show()


    def MC_eps(self):
        eps = 0.1
        policy = np.ones((len(self.states),len(self.actions)))/len(self.actions)
        action_value_fn = np.zeros((len(self.states),len(self.actions)))
        count = np.zeros((len(self.states),len(self.actions)))
        state_value_0=[]
        state_value_1=[]
        state_value_2=[]
        state_value_3=[]
        state_value_4=[]
        state_value_5=[]
        start_time = time.time()
        for episode in range(self.num_episodes):
            state = self.states[2]
            action_index = random.randint(0,len(self.actions)-1)

            G = 0
            done = False
            states=[]
            actions=[]
            rewards=[]
 
            while not done:
                states.append(state)
                actions.append(action_index)

                prob = np.random.random()
                if prob <= 0.8:
                    next_state = self.states[state+self.actions[action_index]]
                elif prob >0.8 and prob <=0.95:
                    next_state = state
                elif prob >0.95:
                    next_state = self.states[state-self.actions[action_index]] #Stochastic
                rew = self.rewards[next_state]
                rewards.append(rew)

                state = next_state
                action_index = np.random.choice(range(len(self.actions)),size=1,p=policy[state,:])[0]
                if next_state == 5 or next_state == 0:
                    done = True
            # print("states",states)
            # print("actions",actions)
            # print("rewards",rewards)
        
            G = 0
            T = len(states)
            
            for t in reversed(range(T)):
                G = self.gamma*G + rewards[t]
                found = False
                for i in range(t):
                    if states[i] == states[t] and actions[i] == actions[t]:
                        found = True
                if not found:
                    count[states[t]][actions[t]] += 1
                    action_value_fn[states[t]][actions[t]] += (G - action_value_fn[states[t]][actions[t]])/count[states[t]][actions[t]]

                    optimal_action_index = np.argmax(action_value_fn[states[t],:]) 
                    
                    policy[states[t],:] = eps/len(self.actions)
                    policy[states[t],optimal_action_index] = 1 - eps + eps/len(self.actions)
            state_value_0.append(policy[0][0]*action_value_fn[0][0] + policy[0][1]*action_value_fn[0][1])
            state_value_1.append(policy[1][0]*action_value_fn[1][0] + policy[1][1]*action_value_fn[1][1])
            state_value_2.append(policy[2][0]*action_value_fn[2][0] + policy[2][1]*action_value_fn[2][1])
            state_value_3.append(policy[3][0]*action_value_fn[3][0] + policy[3][1]*action_value_fn[3][1])
            state_value_4.append(policy[4][0]*action_value_fn[4][0] + policy[4][1]*action_value_fn[4][1])
            state_value_5.append(policy[5][0]*action_value_fn[5][0] + policy[5][1]*action_value_fn[5][1])
        total_time = time.time() - start_time
        print('Time taken for Monte Carlo Epsilon Soft to converge is ',total_time)

        # print(self.policy) 
        # print(self.action_value_fn)

        fig = plt.figure()
        fig.suptitle("Epsilon Soft Stochastic")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_0,label = "State 0")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_1,label = "State 1")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_2,label = "State 2")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_3,label = "State 3")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_4,label = "State 4")
        plt.plot(list(range(1,self.num_episodes+1)),state_value_5,label = "State 5")
        plt.legend()
        plt.xlabel("Number of episodes")
        plt.ylabel("State values")
        plt.show()

if __name__ == "__main__":
    env_obj = algorithms()
    