-----------------------------------------------------------------------------------
CODE COMPONENTS:

- Prediction Algorithms (By Yuhang Lin, Hari Krishna Majety)
- Infrastructure (By Srinivas Parasa)

Prediction Algorithms:
- Executed in AWS DeepLearning instances.
- Requires Anaconda with Tensorflow, Tensorflow-gpu, apscheduler packages
- Deep Q Network Models by Yuhang Lin
- Policy Gradient + Baseline Models by Hari Krishna Majety

Infrastructure:



-----------------------------------------------------------------------------------
STEPS TO SETUP THE PROJECT:

Prediction Instance:
-Clone the git repository() to AWS Deep Learning Instance and install the required libraries
-Make sure the infrastructure modules are all up and set the usage info ip address in the file apis/getcpuinfo.py and the scaling API ip address as parameter in the respective prompt.
-For using Policy Gradients (2 variants) & baseline models(2 variants) using pretrained weights,  run the algorithm/MainPredict.py and enter the choice to model to use in the respective prompt.
-To train new models, run
---RLWithLSTM.py for LSTM variant of Policy Gradients based reinforcement learning
---RLWithDNN3Output.py for DNN variant of Policy Gradients based reinforcement learning
---SimpleLSTM.py for LSTM variant of baseline models
---3Output.py for DNN variant of baseline models
-----All these models require a csv file containing containers' CPU usage data sampled continuously during a certain interval of time. Each row(i) in usageData.csv represents CPU usage data of all containers at i-th time instant.


Infrastructure Instance:


-----------------------------------------------------------------------------------
FILES BY Hari Krishna Majety:

Main Files:
-MainPredict.py
-RLWithLSTM.py
-RLWithDNN3Output.py
-SimpleLSTM.py
-3Output.py

Other Development related files:
3OutputDNNV2BestWeights.hdf5  RLWithDNN3Output.ipynb    
3Output.ipynb                 LSTMV3RLBestWeights.hdf5  
MainPredict.ipynb             RLWithLSTM.ipynb
3OutputRLV4BestWeights.hdf5   SimpleLSTMV4BestWeights.hdf5
SimpleLSTM.ipynb

All ipynb notebooks were used for experimentation with various model architectures and hdf5 files store the weights after training the models.

-----------------------------------------------------------------------------------