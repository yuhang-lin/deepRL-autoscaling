from __future__ import division
import requests
import json
import time

def cpuinfo():

    headers = { 
        'Content-type': 'application/json',
    }   



    conlist=[]

    it=0
    conlist.append('rubis-wg-0')
    conlist.append('rubis-control-tier-0')
    conlist.append('vmtouch-rubis-db-data-0')
    conlist.append('rubis-web-tier-0')
    conlist.append('rubis-db-tier-0')



    # for each container get the current cpu time
    it=0

    cpudata=[]

    while it<len(conlist):
        req='http://152.46.19.96:8080/api/v2.0/stats/'+str(conlist[it])+'?type=docker&count=2'

        response = requests.get(req, headers=headers)

        coninfo=json.loads(response.content.decode('utf-8'))

        #get cpu info now
        contid=str(conlist[it])

        #get keys in this dictionary
        dickeys=list(coninfo.keys())

        cpu_percentage=coninfo[dickeys[0]][1]['cpu_inst']['usage']['total']/(10**7)

        cpudata.append(cpu_percentage)

        it=it+1
    return cpudata


def main():
    print("Hello world")
    cpuinfo()

if __name__== "__main__":
  main()
