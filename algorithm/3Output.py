#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
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
#from getcpuinfo import cpuinfo
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
from keras import regularizers


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


# ### Set the Target Labels

# In[3]:


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


# In[4]:


cpuUsageData = np.expand_dims(np.array(cpuUsage,dtype = 'float'),axis = 2)
cpuScaling = np.array(target,dtype = 'float')
print(cpuScaling.shape)
print(cpuUsageData.shape)

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)


# ### Preprocessing Input Data

# In[5]:


'''cpuUsage = []
cpuScaling = []
historyParam = 50
index = historyParam

while index<len(inpData):
    currRec = [[inpData[i]] for i in range(index-historyParam,index)]
    cpuUsage.append(currRec)
    index+=1'''
    


# In[6]:


'''cpuScaling = target[historyParam-1:len(target)]


cpuUsageData = np.array(cpuUsage,dtype = 'float')
cpuScaling = np.array(cpuScaling,dtype = 'float')

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)'''


# In[7]:


#len(inpData)
X_train, X_test, y_train, y_test = train_test_split(cpuUsageData , cpuScalingCategorical, test_size=0.25, random_state=42)


# In[8]:


cpuUsageData.shape
#np.shape(cpuUsage)


# In[9]:


cpuScalingCategorical.shape


# In[10]:


model = Sequential()
#model.add(SpatialDropout1D(0.3))
model.add(Flatten(batch_input_shape = (None,50,1)))

model.add(Dense(50, activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(25, activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))

model.add(Dropout(0.3, noise_shape=None, seed=None))

#model.add(SpatialDropout1D(0.3))
model.add(Dense(12,activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(6,activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))

model.add(Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(3, activation='softmax',kernel_initializer='he_normal'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
#model.summary()


# In[11]:


model.summary()


# In[12]:


filepath="3OutputDNNV2BestWeights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[13]:


'''if os.path.isfile("3OutputDNNBestWeights.hdf5"):
    #print("Hey")
    model.load_weights("3OutputDNNBestWeights.hdf5")'''


# In[15]:


history = model.fit(X_train, y_train, epochs = 25,validation_data = [X_test, y_test],verbose = 1, callbacks = callbacks_list,batch_size = 1000)

