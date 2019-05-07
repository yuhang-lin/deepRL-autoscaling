#!/usr/bin/env python
# coding: utf-8

# # Container Auto-scaling with Deep Reinforcement Learning

# by Yuhang Lin

# # 1. Defining the approach 

# In[120]:


import numpy as np
import datetime
import keras
import csv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### first load the data

# In[121]:


cpu_list = []
time_list = []
window = 50
max_threshold = 70
min_threshold = 40
with open('cdata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            if line_count < 10:
                print(f'At time {row[0]}: {row[1]} percent of CPU was used.')
            line_count += 1
            time_list.append(row[0])
            cpu_list.append(float(row[1]))
    print(f'Processed {line_count} lines.')


# In[122]:


correct_action = []
index = 0
while index + 1 < len(cpu_list):
    if cpu_list[index + 1] > max_threshold:
        correct_action.append(1)
    elif cpu_list[index + 1] < min_threshold:
        correct_action.append(0)
    else:
        correct_action.append(2)
    index += 1


# # 2. Modeling the Network

# In[123]:


# simple fully connected layer model 
# with 200 hidden units in first layer
# and 1 sigmoid output
inputs = keras.layers.Input(shape=(1, window))
flattened_layer = keras.layers.Flatten()(inputs)
full_connect_1 = keras.layers.Dense(units=24,activation='relu',use_bias=False,)(flattened_layer)
full_connect_2 = keras.layers.Dense(units=12,activation='relu',use_bias=False,)(full_connect_1)
full_connect_3 = keras.layers.Dense(units=6,activation='relu',use_bias=False,)(full_connect_2)
sigmoid_output = keras.layers.Dense(1,activation='sigmoid',use_bias=False)(full_connect_3)
policy_network_model = keras.models.Model(inputs=inputs,outputs=sigmoid_output)
policy_network_model.summary()


# # 3. Defining loss 

# In[125]:


episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')


# In[126]:


def m_loss(episode_reward):
    def loss(y_true,y_pred):
        # feed in y_true as actual action taken 
        # if actual action was up, we feed 1 as y_true and otherwise 0
        # y_pred is the network output(probablity of taking up action)
        # note that we dont feed y_pred to network. keras computes it
        
        # first we clip y_pred between some values because log(0) and log(1) are undefined
        tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(y_pred)
        # we calculate log of probablity. y_pred is the probablity of taking up action
        # note that y_true is 1 when we actually chose up, and 0 when we chose down
        # this is probably similar to cross enthropy formula in keras, but here we write it manually to multiply it by the reward value
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(tmp_pred)
        # multiply log of policy by reward
        policy_loss=keras.layers.Multiply()([tmp_loss,episode_reward])
        return policy_loss
    return loss


# In[127]:


episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')
policy_network_train = keras.models.Model(inputs=[inputs,episode_reward],outputs=sigmoid_output)

my_optimizer = keras.optimizers.RMSprop(lr=0.001)
policy_network_train.compile(optimizer=my_optimizer,loss=m_loss(episode_reward),)


# # 4. Reward Engineering

# In[129]:


def get_reward(action, current_index):
    '''
    Reward function for the policy.
    '''
    if cpu_list[current_index] > max_threshold and action == 0:
        return -1
    if cpu_list[current_index] < min_threshold and action == 1:
        return -1
    return 1


# In[130]:


def generate_episode(policy_network):
    states_list = [] # shape = (x,80,80)
    up_or_down_action_list=[] # 1 if we chose up. 0 if down
    rewards_list=[]
    network_output_list=[]
    policy_output_list = []
    start = 0
    while start + window < len(cpu_list):
        network_input = np.array(cpu_list[start:start + window]) / 100
        processed_network_input = np.expand_dims(network_input,axis=0) 
        states_list.append(processed_network_input)
        reshaped_input = np.expand_dims(processed_network_input,axis=0) 
        up_probability = policy_network.predict(reshaped_input,batch_size=1)[0][0]
        network_output_list.append(up_probability)
        policy_output_list.append(up_probability)
        actual_action = 3 - np.random.choice(a=[2,3],size=1,p=[up_probability,1-up_probability]) # 2 is up. 3 is down
        action_value = actual_action[0]
        up_or_down_action_list.append(action_value)
        reward = get_reward(action_value, start)
        rewards_list.append(reward)
        start += 1
    return states_list,up_or_down_action_list,rewards_list,network_output_list


# the function is plain and simple, nothing to discuss about it.
# so lets play 1 game and see what happens

# In[131]:


states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)


# In[132]:


print("length of states= "+str(len(states_list)))# this is the number of frames
print("shape of each state="+str(states_list[0].shape))
print("length of rewards= "+str(len(rewards_list)))


# In[133]:


# lets see sample of policy output
print(network_output_list[30:50]) 


# since the network is not trained, its output is about 50% all time. meaning . that it does not know which action is better now and outputs a probablity of about 0.5 for all states

# In[134]:


#lets see a sample what we actually did: 1 means we went up, 0 means down
up_or_down_action_list[30:50]


# In[135]:


# lets see sample of rewards
print(rewards_list[50:100]) 


# In[136]:


# lets see how many times we won through whole game:
print("count win="+str(len(list(filter(lambda r: r>0,rewards_list)))))
print("count lose="+str(len(list(filter(lambda r: r<0,rewards_list)))))
print("count zero rewards="+str(len(list(filter(lambda r: r==0,rewards_list)))))


# # 5. Example of simluation and training

# In[142]:


# first generate an episode:
states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)


# In[143]:


print("length of states= "+str(len(states_list)))# this is the number of frames
print("shape of each state="+str(states_list[0].shape))
print("length of rewards= "+str(len(rewards_list)))


# In[144]:


#preprocess inputs for training: 
    
x=np.array(states_list)

#episode_reward=np.expand_dims(process_rewards(rewards_list),1)
episode_reward=np.expand_dims(rewards_list,1)
y_tmp = np.array(up_or_down_action_list) # 1 if we chose up, 0 if down
y_true = np.expand_dims(y_tmp,1) # modify shape. this is neccassary for keras


print("episode_reward.shape =",episode_reward.shape)
print("x.shape =",x.shape)
print("y_true.shape =",y_true.shape)


# In[145]:


# fit the model with inputs and outputs.
policy_network_train.fit(x=[x,episode_reward],y=y_true)


# # 6. Training the network

# In[146]:


# we define a helper function to create a batch of simulations
# and after the batch simulations, preprocess data and fit the network
def generate_episode_batches_and_train_network(n_batches=10):
    batch_state_list=[]
    batch_up_or_down_action_list=[]
    batch_rewards_list=[]
    batch_network_output_list=[]
    for i in range(n_batches):
        states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)   
        batch_state_list.extend(states_list[15:])
        batch_network_output_list.extend(network_output_list[15:])
        batch_up_or_down_action_list.extend(up_or_down_action_list[15:])
        batch_rewards_list.extend(rewards_list[15:])
    
    episode_reward=np.expand_dims(batch_rewards_list,1)
    x=np.array(batch_state_list)
    y_tmp = np.array(batch_up_or_down_action_list)
    y_true = np.expand_dims(y_tmp,1)
    policy_network_train.fit(x=[x,episode_reward],y=y_true)

    return batch_state_list,batch_up_or_down_action_list,batch_rewards_list,batch_network_output_list


# In[147]:


train_n_times = 31
for i in range(train_n_times):
    states_list,up_or_down_action_list,rewards_list,network_output_list=generate_episode_batches_and_train_network(10)
    if i%10==0:
        print("i="+str(i))
        rr=np.array(rewards_list)
        print('count win='+str(len(rr[rr>0]))) 
        policy_network_model.save("policy_network_model_nn.h5")
        policy_network_model.save("policy_network_model_nn"+str(i)+".h5")
        with open('rews_model_simple.txt','a') as f_rew:
            f_rew.write("i="+str(i)+'       reward= '+str(len(rr[rr > 0])))
            f_rew.write("\n")
print("Training is completed")


# # 7. Loading model from file

# In[149]:


policy_network_model=keras.models.load_model("policy_network_model_nn30.h5")
policy_network_model.summary()


# In[159]:


def evaluate(expected, actual, start):
    counter = 0
    for i in range(start, len(actual)):
        if expected[i] == 2 or expected[i] == actual[i]:
            counter += 1
    return counter


# In[160]:


accuracy_list = []
for i in range(3):
    states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)
    res = evaluate(correct_action, up_or_down_action_list, window)
    accuracy_list.append(res / len(correct_action))
print(sum(accuracy_list) / len(accuracy_list))

