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
-Executed in Infrastructure VCL instance.
-Setup Rubis containers with variable load.
-Expose API for CPU percentage.
-Expose API for Scaling module.


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
-Clone the git repository() to Infrastructure Instance. if you face "git not installed error" i included the command in "anonymouscmd.sh" file with comments.
-Go inside the /deepRL-autoscaling/scripts folder and run "dockersetup.sh" shell script to install docker and docker compose.
-In the same folder follow the steps in "Rubiscontainers.sh" file to up all the four containers. Users are manually required to edit two files please follow them exactly.
-Run "cadvisor_setup.sh" to up cadvisor container which helps to get CPU stats.
-Run "flasksetup.sh" to install required packages for flask API.
-Go to "deepRL-autoscaling/apis" and run "python app.py" which starts API to hit scaling module.

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

FILES BY Srinivas Parasa:
-/deepRL-autoscaling/scripts
 -anonymouscmd.sh
 -dockersetup.sh
 -Rubiscontainers.sh
 -cadvisor_setup.sh
 -flasksetup.sh
-/deepRL-autoscaling/apis
 -getcpuinfo.py
 -app.py
 -test.py
 -cpuinfo
-deepRL-autoscaling/Rubis-Docker/RubisClient/Client/edu/rice/rubis/client/UserSession.java


.sh files are used to setup container infrastructure and .py files are used to expose APIs for scaling and cpupercentage module.
-----------------------------------------------------------------------------------
