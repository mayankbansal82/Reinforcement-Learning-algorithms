import matplotlib.pyplot as plt
import csv
import numpy as np
import time

class environment:
    def __init__(self):
        self.grid = self.load_map('map.csv')
        self.rewards = self.create_rewards(self.grid)
        #plt.imshow(self.rewards,cmap='gray')
        #plt.show() 

        #Deterministic actions
        self.actions = [[[0,1,1]],[[1,1,1]],[[1,0,1]],[[1,-1,1]],[[0,-1,1]],[[-1,-1,1]],[[-1,0,1]],[[-1,1,1]]] 

        #Stochastic actions
        #self.actions = [[[0,1,0.8],[1,1,0.1],[-1,1,0.1]],[[1,1,0.8],[0,1,0.1],[1,0,0.1]],[[1,0,0.8],[1,1,0.1],[1,-1,0.1]],[[1,-1,0.8],[1,0,0.1],[0,-1,0.1]],[[0,-1,0.8],[1,-1,0.1],[-1,-1,0.1]],[[-1,-1,0.8],[0,-1,0.1],[-1,0,0.1]],[[-1,0,0.8],[-1,-1,0.1],[-1,1,0.1]],[[-1,1,0.8],[-1,0,0.1],[0,1,0.1]]]

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
        rewards = rewards - 49*grid
        rewards[7][10] = 100
        return rewards
      
class algorithms:
    def __init__(self,grid,rewards,actions):
        self.nA = len(actions)
        self.value_fn = np.zeros((grid.shape[0],grid.shape[1]))
        self.policy = np.ones((grid.shape[0],grid.shape[1],self.nA))/self.nA
        _ = self.value_iteration(grid,self.value_fn,rewards,actions)
        self.policy_iteration(grid,self.value_fn,rewards,actions)
        self.GPI(rewards,self.value_fn,actions,grid)

    def value_iteration(self,grid,value_fn,rewards,actions,thres = 0.001,gamma = 0.95):
        start_time = time.time()
        delta = thres
        nA = len(actions)
        policy = np.ones((grid.shape[0],grid.shape[1],nA))/nA
        while(delta>=thres):
            delta = 0
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i][j]==0:
                        v = value_fn[i][j]
                        z=[]
                        # z = [prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col]) for [row,col,prob] in actions]
                        for m in range(len(actions)):
                            sum=0
                            for [row,col,prob] in actions[m]:
                                sum += prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col])
                            z.append(sum)
                        value_fn[i][j] = max(z)
                        delta = max(delta,abs(v-value_fn[i][j]))
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j]==0:
                    z=[]
                    # k = np.argmax(np.array([prob*(rewards[i + row][j + col] + gamma*value_fn[i + row][j + col]) for [row,col,prob] in actions]))
                    for m in range(len(actions)):
                        sum = 0
                        for [row,col,prob] in actions[m]:
                            sum += prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col])
                        z.append(sum)
                    k = np.argmax(z)
                    policy[i,j,:]*=0
                    policy[i,j,k]=1 
        total_time = time.time() - start_time
        print('Time taken for Value Iteration to converge is ',total_time)

        fig = plt.figure()
        fig.suptitle("Value Iteration")

        x = np.array([i for i in range(grid.shape[0])]).astype(float)
        y = np.array([j for j in range(grid.shape[1])]).astype(float)
        X, Y = np.meshgrid(x,y)
        u = X.copy()
        v = Y.copy()
        for i in x.astype(int):
            for j in y.astype(int):
                if grid[i][j]==1:
                    u[j,i] = 0
                    v[j,i] = 0
                act_idx = np.argmax(policy[i][j][:])
                u[j,i] = -1*actions[act_idx][0][0]*0.1
                v[j,i] = actions[act_idx][0][1]*0.1
        plt.subplot(211)
        plt.title("Policy")
        plt.quiver(Y,X,v,u,scale=10)
        plt.imshow((1-grid),cmap='gray')
        plt.subplot(212)
        plt.title("Value Function")
        plt.imshow(value_fn,cmap='gray')
        plt.show()
        return policy

    def policy_evaluation(self,policy,grid,value_fn,rewards,actions,thres = 0.001,gamma = 0.95):
        delta = thres
        while(delta>=thres):
            delta=0
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i][j] == 0:
                        add = 0
                        v = value_fn[i][j]
                        z=[]
                        # z = [prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col]) for [row,col,prob] in actions]
                        for m in range(len(actions)):
                            sum=0
                            for [row,col,prob] in actions[m]:
                                sum += prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col])
                            z.append(sum)
                        for k in range(len(z)):
                            add += policy[i,j,k]*z[k]
                        value_fn[i][j] = add 
                        delta = max(delta,abs(v-value_fn[i][j]))
        return value_fn
        
    def policy_improvement(self,policy,grid,value_fn,actions,rewards,gamma = 0.95):
        policy_stable = True
        for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i][j] == 0:
                        z=[]
                        old_action = np.argmax(policy[i,j,:])
                        # k = np.argmax(np.array([prob*(rewards[i + row][j + col] + gamma*value_fn[i + row][j + col]) for [row,col,prob] in actions]))
                        for m in range(len(actions)):
                            sum = 0
                            for [row,col,prob] in actions[m]:
                                sum += prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col])
                            z.append(sum)
                        k = np.argmax(z)
                        policy[i,j,:] = policy[i,j,:]*0
                        policy[i,j,k] = 1
                        if old_action != k:
                            policy_stable = False
        return policy_stable
        
    def policy_iteration(self,grid,value_fn,rewards,actions):
        start_time = time.time()
        nA = len(actions)
        policy = np.ones((grid.shape[0],grid.shape[1],nA))/nA
        policy_stable = False
        while policy_stable == False:
            value_fn = self.policy_evaluation(policy,grid,value_fn,rewards,actions)
            policy_stable = self.policy_improvement(policy,grid,value_fn,actions,rewards)
        total_time = time.time() - start_time
        print('Time taken for Policy Iteration to converge is ',total_time)

        fig = plt.figure()
        fig.suptitle("Policy Iteration")

        x = np.array([i for i in range(grid.shape[0])]).astype(float)
        y = np.array([j for j in range(grid.shape[1])]).astype(float)

        X, Y = np.meshgrid(x,y)

        u = X.copy()
        v = Y.copy()

        for i in x.astype(int):
            for j in y.astype(int):
                if grid[i][j]==1:
                    u[j,i] = 0
                    v[j,i] = 0
                act_idx = np.argmax(policy[i][j][:])
                u[j,i] = -1*actions[act_idx][0][0]*0.1
                v[j,i] = actions[act_idx][0][1]*0.1
        
        plt.subplot(211)
        plt.title("Policy")
        plt.quiver(Y,X,v,u,scale=10)
        plt.imshow((1-grid),cmap='gray')
        plt.subplot(212)
        plt.title("Value Function")
        plt.imshow(value_fn,cmap='gray')
        plt.show()
       

    def GPI(self,rewards,value_fn,actions,grid,gamma=0.95):
        start_time = time.time()
        policy_stable = False
        nA = len(actions)
        policy = np.ones((grid.shape[0],grid.shape[1],nA))/nA
        while policy_stable == False:
            policy_stable = True
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i][j] == 0:
                        add = 0
                        v = value_fn[i][j]
                        z = []

                        # z = [prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col]) for [row,col,prob] in actions]

                        for m in range(len(actions)):
                            sum=0
                            for [row,col,prob] in actions[m]:
                                sum += prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col])
                            z.append(sum)

                        for k in range(len(z)):
                            add += policy[i,j,k]*z[k]
                        value_fn[i][j] = add
        
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i][j] == 0:
                        z = []
                        old_action = np.argmax(policy[i,j,:])

                        # k = np.argmax(np.array([prob*(rewards[i + row][j + col] + gamma*value_fn[i + row][j + col]) for [row,col,prob] in actions]))

                        for m in range(len(actions)):
                            sum = 0
                            for [row,col,prob] in actions[m]:
                                sum += prob*(rewards[i + row,j + col] + gamma*value_fn[i + row,j + col])
                            z.append(sum)
                        
                        k = np.argmax(z)

                        policy[i,j,:] = policy[i,j,:]*0
                        policy[i,j,k] = 1
                        if old_action != k:
                            policy_stable = False
        total_time = time.time() - start_time
        print("Time taken for Generalized Policy Iteration to converge is ",total_time)
        fig = plt.figure()
        fig.suptitle("Generalized Policy Iteration")

        x = np.array([i for i in range(grid.shape[0])]).astype(float)
        y = np.array([j for j in range(grid.shape[1])]).astype(float)

        X, Y = np.meshgrid(x,y)
        u = X.copy()
        v = Y.copy()

        for i in x.astype(int):
            for j in y.astype(int):
                if grid[i][j]==1:
                    u[j,i] = 0
                    v[j,i] = 0
                act_idx = np.argmax(policy[i][j][:])
                u[j,i] = -1*actions[act_idx][0][0]*0.1
                v[j,i] = actions[act_idx][0][1]*0.1
        
        plt.subplot(211)
        plt.title("Policy")
        plt.quiver(Y,X,v,u,scale=10)
        plt.imshow((1-grid),cmap='gray')
        plt.subplot(212)
        plt.title("Value Function")
        plt.imshow(value_fn,cmap='gray')
        plt.show()

if __name__ == "__main__":
    env_obj = environment()
    alg_obj = algorithms(env_obj.grid,env_obj.rewards,env_obj.actions)
    







    
    
