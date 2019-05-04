#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries

# In[1]:


import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import LSTM, SpatialDropout1D 
from sklearn.model_selection import train_test_split
import csv
import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import accuracy_score
import sys


# ### Load Raw Training Data

# In[2]:


inpData = []
labels = []

'''
with open('cdata.csv') as csv_file:
    lines = csv.reader(csv_file, delimiter=',')
    for row in lines:
        labels.append(row[0])
        inpData.append(float(row[1]))
'''
inpDataArray = np.loadtxt("usageData.csv",delimiter = ',') 

    


# In[3]:


inpDataArray.shape


# ### Set the Target Labels

# In[4]:


minThreshold = 30
maxThreshold = 70
target = []
cpuUsage = []
historyParam = 50
for i in range(0,inpDataArray.shape[1]):
    inpData = inpDataArray[:,i]
    index = historyParam
    while index < len(inpData):
        currRec = inpData[index-historyParam:index]
        if inpData[index] > maxThreshold:# Scale Up
            target.append(1)
        elif inpData[index] < minThreshold: # Scale Down
            target.append(-1)
        else:
            target.append(0)  #Remain same
        cpuUsage.append(currRec)
        index += 1


# In[5]:


cpuUsageData = np.expand_dims(np.array(cpuUsage,dtype = 'float'),axis = 2)
cpuScaling = np.array(target,dtype = 'float')
print(cpuScaling.shape)
print(cpuUsageData.shape)

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)


# ### Preprocessing Input Data

# In[6]:


'''cpuUsage = []
cpuScaling = []
historyParam = 50
index = historyParam

while index<len(inpData):
    currRec = [[inpData[i]] for i in range(index-historyParam,index)]
    cpuUsage.append(currRec)
    index+=1'''
    


# In[7]:


'''cpuScaling = target[historyParam-1:len(target)]


cpuUsageData = np.array(cpuUsage,dtype = 'float')
cpuScaling = np.array(cpuScaling,dtype = 'float')

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)'''


# In[8]:


#len(inpData)
X_train, X_test, y_train, y_test = train_test_split(cpuUsageData , cpuScalingCategorical, test_size=0.10, random_state=42)


# In[9]:


cpuUsageData.shape
#np.shape(cpuUsage)


# In[10]:


import pandas
testrec = 5
print(cpuScalingCategorical[testrec])
print(cpuScaling[testrec])
print(set(cpuScaling))
myseries=pandas.Series(cpuScaling)
print(myseries.value_counts())


# In[11]:


usageInputLayer = keras.layers.Input(shape=(historyParam,1))
#flattened_layer = keras.layers.Flatten()(usageInputLayer)
LSTM_1 = keras.layers.LSTM((50), return_sequences = True)(usageInputLayer)
full_connect_2 = keras.layers.Dense(25, activation = 'relu',kernel_initializer='he_normal')(LSTM_1)
LSTM_3 = keras.layers.LSTM(25, return_sequences = False)(full_connect_2)
full_connect_3 = keras.layers.Dense(12, activation = 'relu',kernel_initializer='he_normal')(LSTM_3)
full_connect_4 = keras.layers.Dense(6, activation = 'relu',kernel_initializer='he_normal')(full_connect_3)
softmax_output = keras.layers.Dense(3,activation='softmax',kernel_initializer='he_normal')(full_connect_4)
predictionModel = keras.models.Model(inputs=usageInputLayer,outputs=softmax_output)
predictionModel.summary()


# In[12]:


def customLoss(rewardInputLayer):
    def loss(y_true,y_pred):
        tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(y_pred)
        tmp_loss = keras.losses.categorical_crossentropy(y_true, tmp_pred)
        policy_loss=keras.layers.Multiply()([tmp_loss,rewardInputLayer])
        #policy_loss = tf.reduce_sum(rewardInputLayer,axis =-1)
        return policy_loss
    return loss


# In[13]:


rewardInputLayer = keras.layers.Input(shape=(1,),name='rewardInputLayer')
TrainingModel = keras.models.Model(inputs=[usageInputLayer,rewardInputLayer],outputs=softmax_output)
TrainingModel.compile(optimizer="adam",loss=customLoss(rewardInputLayer))


# In[14]:


TrainingModel.summary()


# In[16]:


def generateReward(usage,action,minThreshold,maxThreshold):
    if usage[49] > maxThreshold :
        if action == 0:
            return 1
        elif action == -1:
            return 2
    elif usage[49] < minThreshold: 
        if action == 0:
            return 1
        elif action == 1:
            return 2
    else:
        if action != 0:
            return 1
    return -1


def generateReward2(requiredAction,action):
    predDict = {0:0,1:1,2:-1}
    actualActions = predDict[np.argmax(requiredAction)]
    if(actualActions == action):
        reward = 1
    else:
        reward = -1
    return reward
    
def simulateSituation(pModel,tModel,usageArray,minThreshold,maxThreshold):
    predDict = {0:0,1:1,2:-1}
    #modelInput = np.expand_dims(usageArray,axis=0)
    res = pModel.predict(usageArray)
    #mapper = lambda x: predDict[x]
    actionPredictedOneHot = to_categorical(np.argmax(res,axis = 1),num_classes = 3)
    actionPredicted = np.array([predDict[np.argmax(x)] for x in res])
    rewardsList = []
    #print(actionPredicted," \n",usageArray[0])
    print(len(usageArray))
    for i in range(0,len(usageArray)):
        #actionReward = generateReward(y_train[i],actionPredicted[i],minThreshold,maxThreshold,)
        actionReward = generateReward2(y_train[i],actionPredicted[i])
        if actionReward == None:
            print(actionPredicted[i]," \n",usageArray[i]," ", minThreshold," ",maxThreshold)
            sys.exit()
            
        rewardsList.append(actionReward)
    rewardsArray = np.array(rewardsList)
    
    actionSeries = pandas.Series(actionPredicted)
    print(actionSeries.value_counts())
    rewardSeries = pandas.Series(rewardsList)
    print(rewardSeries.value_counts())
    
    #print(rewardsArray.shape," ",rewardsArray[0])
    #if os.path.isfile("LSTMRLBestWeights.hdf5"):
    #    print("Loading Previous weights!")
    #    tModel.load_weights("LSTMRLBestWeights.hdf5")
    history = tModel.fit(x = [usageArray,rewardsArray], y = actionPredictedOneHot,epochs = 1,batch_size = 1000)
    #print(usageArray[0]," ", actionPredicted," \n", rewardsArray)
    return pModel,tModel,res
    


# In[17]:


#filepath="LSTMRLBestWeights.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

epochs = 10
for i in range(epochs):
    _,_,modelOutputOneHot = simulateSituation(predictionModel,TrainingModel,X_train,30,70)
    #simulateSituation(predictionModel,TrainingModel,X_train,30,70)
    res = predictionModel.predict(X_test)
    print(accuracy_score(np.argmax(modelOutputOneHot,axis = 1),np.argmax(y_train,axis = 1)))
    print(accuracy_score(np.argmax(res,axis = 1),np.argmax(y_test,axis = 1)))


# In[18]:


predictionModel.save("LSTMV3RLBestWeights.hdf5")


# In[ ]:


if os.path.isfile("LSTMRLBestWeight.hdf5"):
    print("Loading Previous weights!")
    TrainingModel.load_weights("LSTMRLBestWeight.hdf5")
res = predictionModel.predict(X_train)


# In[ ]:


accuracy_score(np.argmax(res,axis = 1),np.argmax(y_train,axis = 1))


# In[ ]:


print(to_categorical(np.argmax(y_train,axis = 1),num_classes = 3)[:10,:])
print(y_train[:10,:])


# In[ ]:


np.argmax([[0,0,1],[1,0,0]],axis = 1)


# In[ ]:


temp1 = np.array([[]])


# In[ ]:


X_train[0].shape


# In[ ]:


filepath="LSTMBestWeights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[ ]:


if os.path.isfile("LSTMBestWeights.hdf5"):
    print("Loading Previous weights!")
    model.load_weights("LSTMBestWeights.hdf5")


# In[ ]:


history = model.fit(X_train, y_train, epochs = 15,validation_data = [X_test, y_test], callbacks = callbacks_list,verbose=0)


# In[ ]:


X_test.shape
np.expand_dims(X_test[0],axis = 0).shape


# In[ ]:


res = model.predict(np.expand_dims(X_test[0],axis=0))


# In[ ]:


print(res)
print(np.argmax(res[0]))


# In[ ]:


np.argmax([0,1,0],axis = 0)


# In[ ]:


np.argmax(res,axis = 1)


# In[ ]:


print(inpData[0+1])

