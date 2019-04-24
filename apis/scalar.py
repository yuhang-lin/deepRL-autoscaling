#this code helps to balance the cpu of the containers based on prediction
#of the algorithm.
import os
import pickle

def setVCPU():
	
	predictions=[0,1,1,1]

	underProvisionedContainers = []

	overProvisionedContainers = []
    
	rightlyProvisionedContainers = []

	conlist=[]

	conlist.append('rubis-wg-0')
	conlist.append('rubis-control-tier-0')
	conlist.append('rubis-web-tier-0')
	conlist.append('rubis-db-tier-0')
    
	availableVCPU = 0

      	#read the current vcpu info from the file
	with open('./cpuinfo', 'rb') as f:
		print("reading data from file")
		vcpu_info = pickle.load(f)

	for id in vcpu_info:
		print(id)


	print("read the data from file")

    
        #keep the conatiners into specific list depending on preditions list
	for i in range(0,len(predictions)):
		if predictions[i]==-1:
			overProvisionedContainers.append(i)
		elif predictions[i]==0: 
			rightlyProvisionedContainers.append(i)
		else:
			underProvisionedContainers.append(i)
         
         

        #only execute the algorithm when atleast one is over provisioned and not all
	if len(underProvisionedContainers)>0 and len(underProvisionedContainers)!=4:
		for id in overProvisionedContainers:
			availableVCPU=availableVCPU+vcpu_info[id]*(0.1)
			vcpu_info[id]=vcpu_info[id]*(0.9)


		if availableVCPU==0:
			print("enetered rightly provisioned")
			for id in rightlyProvisionedContainers:
				availableVCPU=availableVCPU+vcpu_info[id]*(0.05)
				vcpu_info[id]=vcpu_info[id]-vcpu_info[id]*(0.05)

		ecpushare=availableVCPU/(len(underProvisionedContainers))
		print(ecpushare)	
                #now assign this cpu to underprovisioned ones
		for id in underProvisionedContainers:
			vcpu_info[id]=vcpu_info[id]+ecpushare


                #now iterate through the vcpu_info array and assign them
		print("new vcpu")
		for id in range(0,len(vcpu_info)):
			print(vcpu_info[id])
			os.system('sudo docker update --cpus '+ str(round(vcpu_info[id],4))+ ' '+ str(conlist[id]))

                #send the new vcpu_info to persistent storage
		with open('./cpuinfo', 'wb') as f:
			pickle.dump(vcpu_info, f)
	else:
		print("all are under or no under")

def main():
	setVCPU()
  
if __name__== "__main__":
	main()

