import pickle

vcpu_info=[0.5,0.5,0.5,0.5]

with open('./cpuinfo', 'wb') as f:
    pickle.dump(vcpu_info, f)

