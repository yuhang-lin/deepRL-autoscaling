#Written: Srinivas Parasa
#Usage: to change VCPU to required values. modify the array below
import pickle

vcpu_info=[1,1,1,1]

with open('./cpuinfo', 'wb') as f:
    pickle.dump(vcpu_info, f)

