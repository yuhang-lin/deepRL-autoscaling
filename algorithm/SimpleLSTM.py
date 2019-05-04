#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries

# In[25]:


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

# In[26]:


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

# In[27]:


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


# In[28]:


cpuUsageData = np.expand_dims(np.array(cpuUsage,dtype = 'float'),axis = 2)
cpuScaling = np.array(target,dtype = 'float')
print(cpuScaling.shape)
print(cpuUsageData.shape)

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)


# ### Preprocessing Input Data

# In[29]:


'''cpuUsage = []
cpuScaling = []
historyParam = 50
index = historyParam

while index<len(inpData):
    currRec = [[inpData[i]] for i in range(index-historyParam,index)]
    cpuUsage.append(currRec)
    index+=1'''
    


# In[30]:


'''cpuScaling = target[historyParam-1:len(target)]


cpuUsageData = np.array(cpuUsage,dtype = 'float')
cpuScaling = np.array(cpuScaling,dtype = 'float')

cpuScalingCategorical = to_categorical(cpuScaling, num_classes = 3)'''


# In[31]:


#len(inpData)
X_train, X_test, y_train, y_test = train_test_split(cpuUsageData , cpuScalingCategorical, test_size=0.10, random_state=3)


# In[32]:


cpuUsageData.shape
#np.shape(cpuUsage)


# In[33]:


import pandas
testrec = 5
print(cpuScalingCategorical[testrec])
print(cpuScaling[testrec])
print(set(cpuScaling))
myseries=pandas.Series(cpuScaling)
print(myseries.value_counts())


# In[34]:


model = Sequential()
#model.add(SpatialDropout1D(0.3))
model.add(LSTM((50),dropout = 0.3,batch_input_shape = (None,50,1), return_sequences = True))
model.add(Dense(25, activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))
model.add(LSTM(25,return_sequences = False))
model.add(Dense(12, activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(rate = 0.3))
model.add(Dense(6, activation = 'relu',
                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
#model.summary()


# In[35]:


model.summary()


# In[36]:


filepath="SimpleLSTMV4BestWeights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[37]:


'''if os.path.isfile("LSTMBestWeights.hdf5"):
    #print("Hey")
    model.load_weights("LSTMBestWeights.hdf5")'''


# In[38]:


history = model.fit(X_train, y_train, epochs = 25,validation_data = [X_test, y_test], callbacks = callbacks_list,verbose=1, batch_size = 1000)


# In[ ]:


# X_test.shape
# np.expand_dims(X_test[0],axis = 0).shape


# # In[ ]:


# res = model.predict(np.expand_dims(X_test[0],axis=0))


# # In[ ]:


# print(res)
# print(np.argmax(res[0]))


# # In[ ]:


# np.argmax([0,1,0],axis = 0)


# # In[ ]:


# np.argmax(res,axis = 1)


# # In[ ]:


# print(inpData[0+1])

