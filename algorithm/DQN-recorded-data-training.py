#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras
import csv
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score


# ### Load Raw Training Data

# In[23]:


inpData = []
with open('usageData.csv') as csv_file:
    lines = csv.reader(csv_file, delimiter=',')
    for row in lines:
        inpData.append(float(row[0]))     


# ### Set the Target Labels

# In[24]:


minThreshold = 30
maxThreshold = 70
target = []
index = 0
while index + 1 < len(inpData):
    if inpData[index + 1] > maxThreshold:# Scale Up
        target.append(1)
    elif inpData[index + 1] < minThreshold: # Scale Down
        target.append(-1)
    else:
        target.append(0)  # Remain the same, i.e. doing nothing
    index += 1


# ### Preprocessing Input Data

# In[25]:


cpuUsage = []
cpuScaling = []
historyParam = 50
index = historyParam

while index<len(inpData):
    currRec = [[inpData[i] / 100] for i in range(index-historyParam,index)]
    cpuUsage.append(currRec)
    index+=1


# In[26]:


cpuScaling = target[historyParam-1:len(target)]

cpuUsageData = np.array(cpuUsage,dtype = 'float')
cpuScaling = np.array(cpuScaling,dtype = 'float')

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)



newTrainX = cpuUsageData
newTrainY = cpuScalingCategorical
for i in range(1,len(cpuUsageData)):
    scalingVal = np.argmax(cpuScalingCategorical[i])
    if scalingVal == 0:
        for j in range(0,7):
            newTrainX = np.append(newTrainX,np.array([cpuUsageData[i]]),axis = 0)
            newTrainY = np.append(newTrainY,np.array([cpuScalingCategorical[i]]),axis = 0)
        
    if scalingVal == 1:
        for j in range(0,10):
            newTrainX = np.append(newTrainX,np.array([cpuUsageData[i]]),axis = 0)
            newTrainY = np.append(newTrainY,np.array([cpuScalingCategorical[i]]),axis = 0)

X_train, X_test, y_train, y_test = train_test_split(newTrainX, newTrainY, test_size=0.30, random_state=21)


# In[30]:


print(len(X_train), len(X_test), len(y_train), len(y_test))
print(cpuScalingCategorical[0])
print(len(cpuScalingCategorical))
print(X_train.shape)


# In[31]:


scalingVal = np.argmax(cpuScalingCategorical[0])
print(cpuScalingCategorical[0])
print(scalingVal)


# In[11]:


# Definition for the agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(18, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_train=True):
        if is_train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# In[12]:


# Definition for the environment
class Environment:
    def __init__(self, data, expected):
        self.counter = 0
        self.window = 50
        self.data = data
        self.expected = expected
        
    def reset(self):
        # return the initial state
        state = self.data[0]
        self.counter = 0
        return state
    
    def get_reward(self, action):
        if np.argmax(self.expected[self.counter - 1]) != action:
            return -1
        return 1

    def step(self, action):
        # use the specified action to get the next state
        self.counter += 1
        done = False
        next_state = self.data[self.counter]
        reward = self.get_reward(action)
        if (self.counter + 1 == len(self.data)):
            done = True
        return next_state, reward, done


# In[13]:


# Settings for training or testing
window = 50
state_size = window
action_size = 3 # 1, 0 and -1

batch_size = 100
EPISODES = 1500


# In[14]:


env = Environment(X_train, y_train)
agent = DQNAgent(state_size, action_size)
agent.load("./save/dqn-4layer.h5")
done = False
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    while not done:
        action = agent.act(state, True)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print("episode: {}/{}, e: {:.2}"
                  .format(e, EPISODES, agent.epsilon))
    if e % 5 == 0:
        agent.save("./save/dqn-4layer.h5")


# In[35]:


# Testing the agent
env = Environment(X_test, y_test)
action_list = []
agent = DQNAgent(state_size, action_size)
agent.load("./save/dqn-4layer30.h5")
done = False
for e in range(1):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    while True:
        action = agent.act(state, False)
        action_list.append(action)
        if done:
            break
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

