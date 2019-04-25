from flask import Flask
import os
import pickle
from flask import request

app = Flask(__name__)
#app.run(host= '0.0.0.0')

@app.route("/changeVCPU",methods=['POST'])
def setVCPU():

	predictions=list(request.json['predictions'])

	underProvisionedContainers = []

	overProvisionedContainers = []

	rightlyProvisionedContainers = []

	conlist=[]

	conlist.append('rubis_rubisclient_1')
	conlist.append('rubis_rubisweb_1')
	conlist.append('rubis_rubis_1')
	conlist.append('rubis_rubisdb_1')

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

	
	return "Sucess"

if __name__ == '__main__':
	app.run(host= '0.0.0.0')
