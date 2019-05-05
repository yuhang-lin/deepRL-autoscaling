#written: SRINIVAS PARASA

#usage: Helps to create 4 containers with variable load



#run all commands at deepRL-autoscaling/Rubis-Docker folder


#command to run rubisdb
sudo docker-compose -p rubis up -d --no-deps --build --force-recreate rubisdb

#now go to "deepRL-autoscaling/Rubis-Docker/Rubis/servlets/context.xml" file 
#on line 34 change url="jdbc:mysql://(RubisDBConatinerID):3306/rubis?autoReconnect=true&amp;useSSL=false"/> insert rubisdb containerID at RubisDBConatinerID. Ex: url="jdbc:mysql://3d658346adac:3306/rubis?autoReconnect=true&amp;useSSL=false"/>



#command to run rubis container
sudo docker-compose -p rubis up -d --no-deps --build --force-recreate rubis


#command to run rubisweb container
sudo docker-compose -p rubis up -d --no-deps --build --force-recreate rubisweb


#now go to "deepRL-autoscaling/Rubis-Docker/RubisClient/Client/rubis.properties" file
#on line 2 insert httpd_hostname = (RubisWeb ContainerID).EX:httpd_hostname = 464d86ee8363
#on line 15 insert servlets_server = (RubisServer ContainerID). EX: ervlets_server = 5608f611aa63
#on line 42 insert database_server = (RubisDB containerID). EX:database_server = 3d658346adac


#command to run RubisClient container in benchmark mode which initilises DB. This takes opporximately 2 hours to complete in background
sudo docker-compose -p rubis up -d --no-deps --build --force-recreate rubisclient

#now go to "deepRL-autoscaling/Rubis-Docker/docker-compose.yml" file"
#on line 42 change "benchmark" to "emulator"

#command to run Rubisclient in emulator mode which starts emulating clients.
sudo docker-compose -p rubis up -d --no-deps --build --force-recreate rubisclient

