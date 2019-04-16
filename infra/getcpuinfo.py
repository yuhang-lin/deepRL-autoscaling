from __future__ import division
import requests
import json

headers = {
    'Content-type': 'application/json',
}


response=requests.get('http://localhost:8080/api/v1.3/containers/docker', headers=headers)

response=json.loads(response.content)

conlist=[]

it=0
conlist.append('rubis-wg-0')
conlist.append('rubis-control-tier-0')
conlist.append('vmtouch-rubis-db-data-0')
conlist.append('rubis-web-tier-0')
conlist.append('rubis-db-tier-0')



# for each container get the current cpu time
it=0

while it<len(conlist):
        #http://152.46.18.217:8080/api/v2.0/stats/rubis_rubis_1?type=docker&count=1
        req='http://152.46.18.217:8080/api/v2.0/stats/'+str(conlist[it])+'?type=docker&count=2'
        response = requests.get(req, headers=headers)

        coninfo=json.loads(response.content)

        #get cpu info now
        contid=str(conlist[it])

        #get keys in this dictionary
        dickeys=coninfo.keys()

        cpu_percentage=coninfo[dickeys[0]][1]['cpu_inst']['usage']['total']/(10**7)

        print(cpu_percentage) 
          
        it=it+1
          
print("completed")
