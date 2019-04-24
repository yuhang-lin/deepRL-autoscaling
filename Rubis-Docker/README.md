# Containerized Rubis BenchmarK
This is the Git repo of the Docker images for the Rubis Benchmark.

## Notice: Unfortunately, this project is currently a work in progress. Further testing is required to still fix bugs if there is any. The benchmark source code is mostly hardcoded and making it run is very challenging. 

This project was started with the aim to provide a docker version of the Rubis java Servlets which might be used to test your docker environment.
From an architectural standpoint, each Rubis tier has been separately build to emulate a 3-tier application and provide a flexible environment where you can more easily scale out or scale up containers within each tier.

```
                                                                     |
               |                          |----- rubis servlet ----- |---- rubis DB (master)
               |                          |   (tomcat container c)   |      (Mysql container e)
Rubis Client --|- apache2.4/rubisproxy--- |                          |
(container a)  |   (container b)          |                          |
               |                          |----- rubis servlet ----- |---- Rubis DB (master)
                                              (tomcat container d)   |       (Mysql container f)

                       WEB                        APPLICATION                  DATA
```

Each Tier image is self-contained.
Refer to the folder of each Docker image for more details.

## Getting Started

This repository contains :

```
 Dockerfiles
 ├── docker-compose.yml
 ├── docker-compose.yml.Multi-Primary
 ├── docker-compose.yml.Simple
 ├── LICENSE.md
 ├── README.md
 ├── Rubis
 ├── RubisClient
 ├── RubisDB
 └── RubisWeb
```

### Requirements

*  [Docker Compose](https://docs.docker.com/compose/install)
*  [Docker Engine](https://docs.docker.com/install/)

### Build 

example using the docker cli :

```bash
docker build -t rubisdb ./RubisDB
docker build -t rubis ./Rubis
docker build -t rubisproxy ./RubisWeb
docker build -t rubisclient ./RubisClient
```
example using docker-compose.yml

```bash
docker-compose build rubisdb rubis rubisweb rubisclient
```

### RUN

example using docker cli:

```bash
docker run --name=rubismysql -d rubisdb
docker run -d -rm -p 5002:8080 --name=rubisservlet -d rubis
docker run -d -rm -p 5001:80 -e LB_MEMBER_1=rubisservlet --name=rubisweb -d rubisproxy
docker run -d -rm -p 5000:80 --name=rubisbenchmark -d rubisclient
```

Tail the logs for more information.

```bash
docker exec -it rubisbenchmark tail -f /var/log/supervisor/Rubis.log
```

Wait for the DB initialization and after that emulation starts, go to IP address at port 5000 ``/bench`` and you can see Rubis running.

example using docker-compose.yaml

```yaml
version: '2'
services:
  rubis:
    build:
      context: ./Rubis
    ports:
     - "5001:8080"
    depends_on:
     - rubisdb
  rubisdb:
    build:
      context: ./RubisDB
  rubisweb:
    build:
      context: ./RubisWeb
    ports:
     - "5002:80"
    environment:
     - LB_MEMBER_1=dockerfiles_rubis_1
    depends_on:
     - rubis
  rubisclient:
    depends_on:
     - rubisdb
     - rubis
     - rubisweb
    build:
      context: ./RubisClient
    ports:
     - "5003:80"
    environment:
     - TARGET=emulate
```

```bash
docker-compose up -d rubisweb
```

Verify that Rubis is correctly running at http://your_host_ip:5003/rubis_servlets
Start the Client.

```bash
docker-compose up -d rubisclient
```

#### DEEP DIVE

Above example deploys one container for each tier. Follow other two examples which explain how to scale up your tier containers.

##### SCALE UP APPLICATION TIER

Example of docker-compose.yml

```yaml
version: '2'
services:
  rubis:
    build:
      context: ./Rubis
    depends_on:
     - rubisdb
    labels:
      loadbalancer: "apache"
  rubisdb:
    build:
      context: ./RubisDB
  rubisweb:
    build:
      context: ./RubisWeb
    ports:
     - "5002:80"
    volumes:
     - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
     - rubis
    security_opt:
     - label:disable
  rubisclient:
    depends_on:
     - rubisdb
     - rubis
     - rubisweb
    build:
      context: ./RubisClient
    ports:
     - "5003:80"
    environment:
     - TARGET=initDB
```

At this point you may decide to reduce the max size of the database connections pool (in Rubis/servlets/context.xml) since with two tomcat the number of DB connections will be doubled.


`
<Resource name="jdbc/Rubismysql" auth="Container" type="javax.sql.DataSource"
               *maxTotal="50"* maxIdle="10" maxWaitMillis="30000" removeAbandonedOnBorrow="true"
               username="rubis" password="rubis" driverClassName="com.mysql.jdbc.Driver"
               url="jdbc:mysql://dockerfiles_rubisdb_1:3306/rubis?autoReconnect=true&amp;useSSL=false"/>
`

The client will also need to know which are the Servlet hosts. This is done by editing the RubisClient/Client/rubis.properties file.


`
servlets_server = dockerfiles_rubis_1,dockerfiles_rubis_2
`

Containers name follow this convention : *Projectname_imagename_id*
Where 
* Projectname = your current folder
* Imagename   = the service name in the docker compose
* id          = is an incremental index     

Rebuild the rubis and rubisclient image.

```bash
docker-compose build rubis rubisclient
```

In this example Rubis Servlets containers can be scaled up using the --scale option. The Proxy will automatically discover changes in the environment 
and will update its configuration accordingly.

```bash
docker-compose up --scale rubis=2 -d rubisweb
Creating network "dockerfiles_default" with the default driver
Creating dockerfiles_rubis_1    ... done
Creating dockerfiles_rubis_2    ... done
Creating dockerfiles_rubis_2    ...
Creating dockerfiles_rubisweb_1 ... done

docker-compose ps
         Name                       Command               State               Ports
------------------------------------------------------------------------------------------------
dockerfiles_rubis_1      /usr/bin/supervisord -c /e ...   Up      8080/tcp
dockerfiles_rubis_2      /usr/bin/supervisord -c /e ...   Up      8080/tcp
dockerfiles_rubisdb_1    /usr/bin/supervisord -c /e ...   Up      3306/tcp, 33060/tcp, 33061/tcp
dockerfiles_rubisweb_1   /usr/bin/supervisord -c /e ...   Up      0.0.0.0:5002->80/tcp

```

Go to http://your_host_IP:5002/rubis_servlets/index.html to verify that Rubis is running.

Start the simulation

```bash
docker-compose up --scale rubis=2 -d rubisclient
dockerfiles_rubisdb_1 is up-to-date
dockerfiles_rubis_1 is up-to-date
dockerfiles_rubis_2 is up-to-date
dockerfiles_rubisweb_1 is up-to-date
Creating dockerfiles_rubisclient_1 ... done
```

See clients log.

```bash
docker-compose exec rubisclient tail -f /var/log/supervisor/Rubis.log

Monitoring scp                 : /usr/bin/scp<br>
Monitoring Gnuplot Terminal    : jpeg<br>


Using Servlets version.<br>
Generating 200 users .. Done!
Generating 1000 old items and 32667 active items.
Generating up to 20 bids per item.
Generating 1 comment per item
..............................
Done!
```

Check the requests status at http://your_host_IP:5002/server-status.

Finally go to http://your_host_IP:5003/bench/ to read the simulation results.

##### SCALE UP DATABASE

This example deploys a database cluster with two databases in a master-master (Multy-Primary) mode.

Example using docker-compose.yml

```yaml
Version: '2'
services:
  rubis:
    build:
      context: ./Rubis
    ports:
     - "5001:8080"
    depends_on:
     - rubisdb
  rubisdb:
    build:
      context: ./RubisDB
    environment:
     - CL_MEMBER_1=dockerfiles_rubisdb_1
     - CL_MEMBER_2=dockerfiles_rubisdb_2
  rubisweb:
    build:
      context: ./RubisWeb
    ports:
     - "5002:80"
    environment:
     - LB_MEMBER_1=dockerfiles_rubis_1
    depends_on:
     - rubis
  rubisclient:
    depends_on:
     - rubisdb
     - rubis
     - rubisweb
    build:
      context: ./RubisClient
    ports:
     - "5003:80"
    environment:
     - TARGET=benchmark
```

NOTE: Differently from above we provide the load-balanced hosts as environment variables. (LB_MEMBER_1)
This is one of the alternatives provided by the rubis Web container. See ./RubisWeb for more details.

As in the previous example the database hostnames has to be added to rubis.properties.

```bash
docker-compose up -d --scale rubisdb=2 rubisweb
Creating network "dockerfiles_default" with the default driver
Creating dockerfiles_rubisdb_1 ... done
Creating dockerfiles_rubisdb_2 ... done
```

Verify the database cluter status

``` bash
docker exec -it dockerfiles_rubisdb_1 mysql -proot -e "SELECT * FROM performance_schema.replication_group_members;"
mysql: [Warning] Using a password on the command line interface can be insecure.

+---------------------------+--------------------------------------+-------------+-------------+--------------+
| CHANNEL_NAME              | MEMBER_ID                            | MEMBER_HOST | MEMBER_PORT | MEMBER_STATE |
+---------------------------+--------------------------------------+-------------+-------------+--------------+
| group_replication_applier | 4f0f9dfa-1c8e-11e8-b5d6-0242ac120003 | 172.18.0.3  |        3306 | ONLINE       |
| group_replication_applier | 4f1c04c5-1c8e-11e8-b6dd-0242ac120002 | 172.18.0.2  |        3306 | ONLINE       |
+---------------------------+--------------------------------------+-------------+-------------+--------------+
```

Modify/uncomment the url in Rubis/servlets/context.xml to ue your database cluster in failover mode.

`
url="jdbc:mysql//dockerfiles_rubisdb_1:3306,dockerfiles_rubisdb_2:3306/rubis"/>
`

Check that each container is up and running (this can be done also using docker-compose ps).

```bash
docker ps
CONTAINER ID        IMAGE                  COMMAND                  CREATED             STATUS                    PORTS                       NAMES
7ec68c35e3c2        dockerfiles_rubisweb   "/usr/bin/supervisord"   23 seconds ago      Up 14 seconds             0.0.0.0:5002->80/tcp        dockerfiles_rubisweb_1
8acb37e0478c        dockerfiles_rubis      "/usr/bin/supervisord"   30 seconds ago      Up 27 seconds             0.0.0.0:5001->8080/tcp      dockerfiles_rubis_1
82c8a1d4868d        dockerfiles_rubisdb    "/usr/bin/supervisord"   12 minutes ago      Up 12 minutes (healthy)   3306/tcp, 33060-33061/tcp   dockerfiles_rubisdb_2
79d1ecbfce8b        dockerfiles_rubisdb    "/usr/bin/supervisord"   12 minutes ago      Up 12 minutes (healthy)   3306/tcp, 33060-33061/tcp   dockerfiles_rubisdb_1
```


Check the Tomcat logs.

```
docker exec -it dockerfiles_rubis_1 tail -f /apache-tomcat-8.5.20/logs/catalina.out
28-Feb-2018 14:09:05.347 INFO [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory Deploying web application directory [/apache-tomcat-8.5.20/webapps/host-manager]
28-Feb-2018 14:09:05.486 INFO [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory Deployment of web application directory [/apache-tomcat-8.5.20/webapps/host-manager] has finished in [139] ms
28-Feb-2018 14:09:05.486 INFO [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory Deploying web application directory [/apache-tomcat-8.5.20/webapps/manager]
28-Feb-2018 14:09:05.539 INFO [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory Deployment of web application directory [/apache-tomcat-8.5.20/webapps/manager] has finished in [53] ms
28-Feb-2018 14:09:05.549 INFO [main] org.apache.coyote.AbstractProtocol.start Starting ProtocolHandler ["http-nio-8080"]
28-Feb-2018 14:09:05.575 INFO [main] org.apache.coyote.AbstractProtocol.start Starting ProtocolHandler ["ajp-nio-8009"]
28-Feb-2018 14:09:05.579 INFO [main] org.apache.catalina.startup.Catalina.start Server startup in 8432 ms
=====DOCKERIZED RUBIS======
Wed Feb 28 14:10:12 UTC 2018 WARN: Establishing SSL connection without server's identity verification is not recommended. According to MySQL 5.5.45+, 5.6.26+ and 5.7.6+ requirements SSL connection must be established by default if explicit option isn't set. For compliance with existing applications not using SSL the verifyServerCertificate property is set to 'false'. You need either to explicitly disable SSL by setting useSSL=false, or set useSSL=true and provide truststore for server certificate verification.
```

Start the simulation

```bash
docker-compose up --scale rubisdb=2 -d rubisclient
dockerfiles_rubisdb_1 is up-to-date
dockerfiles_rubisdb_2 is up-to-date
dockerfiles_rubis_1 is up-to-date
dockerfiles_rubisweb_1 is up-to-date
Creating dockerfiles_rubisclient_1 ... done
```


# Built With

* [apache](https://hub.docker.com/r/mysql/mysql-server/)                                    - apache 2.4.
* [mod_proxy_balancer](https://httpd.apache.org/docs/2.4/mod/mod_proxy_balancer.html)       - apache load balancer. 
* [mod_status](https://httpd.apache.org/docs/2.4/mod/mod_status.html)                       - apache status report.
* [MPM](https://httpd.apache.org/docs/2.4/en/mod/worker.html)                               - apache multi threading .

## Versioning

For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## License

This project is licensed under the LGPL License - see the [LICENSE.md](LICENSE.md) file for details

