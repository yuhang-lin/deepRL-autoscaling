#!/bin/bash

############################# DESCRIPTION ########################################
# automatically generates a balancers configuration
# for apache. 
# A balancer Vectors, DEPLOY_ENV, is used to store one variable
# for each backend host.
# Its values are read respectively from:
#  1) LB_MEMBER_* defined in a docker-compose or using the -e option
#  2) Using links (not working with docker compose >= v2)
#  3) Docker API, NOTE: in centos due to selinux
#     restrictions the option "--security-opt label:disable" 
#     is required whether getenforce=enforcing.
##################################################################################

#get the loadbalanced hosts from the Docker API!
function get_backend () {

python << EOF
import docker

client=docker.DockerClient(base_url='unix://var/run/docker.sock',version='auto')
hosts=''
for container in client.containers.list(filters={"label":"loadbalancer=apache"}):
  hosts+=container.id[:12]+' '
print hosts
EOF
}

#write the apache http.conf
function create_conf () {

  local _conf=$1

  cat << EOF > $_conf
   <VirtualHost *:80>
     ServerName $HOSTNAME
     ServerAlias $WEB_NAME
     # increase allowed size of a client's HTTP request-line, required by Rubis GET requests
     LimitRequestLine 64000
     LimitRequestFieldSize 64000
     # set 10 minutes timeout ... large enough to avoid 502
     TimeOut 600

     <Proxy "balancer://mycluster">
EOF
 ## now loop through the above array of hostname, add one balancer member for each tomcat container
    for i in "${DEPLOY_ENV[@]}"
    do
      member=$(eval "echo \$$i")
      echo $member
      echo -e "\tBalancerMember \"http://$member:$PORT_ENV\"">>$_conf
    done

    cat << EOF >> $_conf
       ProxySet lbmethod=bytraffic
     </Proxy>

     ProxyPass "/rubis_servlets/" "balancer://mycluster/rubis_servlets/"
     ProxyPassReverse "/rubis_servlets/" "balancer://mycluster/rubis_servlets/"
   </VirtualHost>
EOF
}

echo "start apache!"


/bin/bash -c '/usr/sbin/httpd -k start'


while true;
do

  ### 1) Env variables
  DEPLOY_ENV=( ${!LB_MEMBER_*} )
  if [ -z "${PORT_ENV}" ]; then
    PORT_ENV=8080
  fi

  ### 2) Links
  if [ -z ${DEPLOY_ENV} ]; then
    DEPLOY_ENV=( $(env | awk -F "=" '{print $1}' | grep '[a-zA-Z]*_[0-9]*_PORT_[0-9]*_TCP_ADDR') )
    PORT_ENV=( $(env | awk -F "=" '{print $1}' | grep '[a-zA-Z]*_[0-9]*_PORT_[0-9]*_TCP_PORT') )
  fi

  ### 3) Docker API
  #ugly workaround - creates a variable per balanced host
  if [  -z "${DEPLOY_ENV}" ]; then

   if [ -z "${PORT_ENV}" ]; then
     PORT_ENV=8080
   fi

   cluster=( $(get_backend) )

   for(( e=0; e<${#cluster[@]}; e++ ));
   do
     declare "LB_MEMBER_$e"="${cluster[$e]}"
   done

   DEPLOY_ENV=( ${!LB_MEMBER_*} )
  fi

  echo ${DEPLOY_ENV[*]}
if [ ! -z "${DEPLOY_ENV}" ]; then

    filename='/etc/httpd/conf.d/rubis.conf'
    filename_2='/etc/httpd/conf.d/rubis_new.conf'
    if [ ! -f $filename ]; then

      create_conf $filename

      echo "restart apache!"
      /bin/bash -c '/usr/sbin/httpd -k restart'

    else
      create_conf $filename_2

      #check for file changes using hashing
      m1=$(md5sum "$filename"| awk '{ print $1 }')
      m2=$(md5sum "$filename_2"| awk '{ print $1 }')

      if [ "$m1" != "$m2" ] ; then
        echo "Config File has changed!"
        rm -f $filename && cp $filename_2 $filename
        /bin/bash -c '/usr/sbin/httpd -k restart'
      else
         echo "nothing to do - no changes required"
      fi
      rm -f $filename_2

      HTTP_PID=$(cat /etc/httpd/run/httpd.pid)
      ps -p $HTTP_PID|grep httpd&>/dev/null

      if [ ! $? ]; then
        exit 1
      else
        echo "httpd is running with PID=$HTTP_PID"
      fi
    fi


  else
    echo "nothing to do, Balancer members are not defined"
    exit 1
  fi
  #
  sleep 10
  unset ${DEPLOY_ENV[*]}
  unset DEPLOY_ENV

done
