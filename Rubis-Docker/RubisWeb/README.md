#  Containerized Rubis web

This is the Git repo of the Docker image for a Rubis Proxy wich uses apache.
This image configure a dynamic load balancer which distributes client requests across a group of Rubis Servlet containers.It reconfigures itself when a balanced member redeploys, joins or leaves.
It is intended to be used in an environment with a single docker host. Use instead Swarm and a load-balanced network if you have cluster of machines running Docker.
[swarm](https://docs.docker.com/get-started/part3/#run-your-new-load-balanced-app)

## Getting Started

This repository contains :

```
RubisWeb
├── cluster
│   └── create_balancers.sh   ( create a dynamic load balancer which forward requests to Rubis Servlet )
├── conf
│   ├── 00-mpm.conf           ( configure the Multi-Processing Module - controls the number of threads which accept HTTP requests )
│   └── 11-mod_status.conf    ( enable monitoring)
├── Dokerfile
├── LICENSE.md
├── README.md
├── ssh-keys                  
│   ├── rubis_rsa_key
│   └── rubis_rsa_key.pub
└── supervisord.conf          (used to manage runnign programms)
```

### Build 

example :
```
docker build -t rubisproxy  .
```

### RUN

This image exposes the ports 80 and 22 for the respectively apache and ssh daemon.

example:

Run the container and Map TCP ports 80 and 22 to TCP port 5005 and 2222 on the host.

```
docker run -p 5005:80 -p 2222:22 -e LB_MEMBER_1=rubistomcat --name=rubisweb -d rubisproxy
```
Go to IP address at port 5005 and you can see Rubis running. If you are interested to 
know how the apache is performing take a look at /server-status to get a full status report.

#### USAGE

You can use Rubis proxy in three different ways:

 *running with Docker legacy links
 *running with environment variables
 *running in self configuring mode

##### Self Configuring Mode
Rubis Proxy is able to self configuring itself on cluster changes. It periodically fetches the Docker API to get the cluser nodes which ahs to be balanced.
In order to fully functioning the docker socket has to be mounted as a volume. Some Linux distributions (CENTOS, RHEL, FEDORA) using selinux requires also the option ``--security_opt:label:disable`` to be provided.

example of docker-compose.yml

```
version: '2'
services:
  rubis:
    image: "rubis/rubis"
    depends_on:
     - rubisdb
    labels:
      loadbalancer: "apache"
  rubisdb:
    image: "rubis/rubisdb"
  rubisweb:
    image: "rubis/rubisproxy" 
    ports:
     - "80:80"
    volumes:
     - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
     - rubis
    security_opt:
     - label:disable
```
Note that the load-balanced cluster need to be labelled ``labels: loadbalancer: "apache"`` in order to rubisproxy to know which are the backend hosts.

##### Legacy Links

example of legacy links
```
docker run --name=rubis -d rubisservlets1
docker run --name=rubis -d rubisservlets2
docker run -d -p 80:80 --link rubisservlets1:rubisservlets1 --link rubisservlets2:rubisservlets2 rubisproxy

```
##### Environment Variables (V>=2.0)

When you create a Apache Server container, you must configure the load balancer by using the -e option or the docker-compose file and specifying one or more of the following environment variables.

     LB_MEMBER_1=container_name_1
     LB_MEMBER_2=container_name_2
     ....
     LB_MEMBER_N=container_name_N

The LB_MEMBER_* variable tells apache which are the backend servers which serve the requests. 
Refer to the name of your Rubis Servlet containers to determine the hostnames of the balanced servers.

Example of docker-compose.yaml v2

```
version: '2'
services:
  rubis:
    image: "rubis/rubis"
    depends_on:
     - rubisdb
    labels:
      loadbalancer: "apache"
  rubisdb:
    image: "rubis/rubisdb"
  rubisweb:
    image: "rubis/rubisproxy"
    ports:
     - "80:80"
    environment:
     - LB_MEMBER_1=dockerfiles_rubis_1
     - LB_MEMBER_1=dockerfiles_rubis_2
     - LB_MEMBER_1=dockerfiles_rubis_3
    depends_on:
     - rubis
```

## Built With

* [apache](https://hub.docker.com/r/mysql/mysql-server/)                                    - apache 2.4.
* [mod_proxy_balancer](https://httpd.apache.org/docs/2.4/mod/mod_proxy_balancer.html)       - apache load balancer. 
* [mod_status](https://httpd.apache.org/docs/2.4/mod/mod_status.html)                       - apache status report.
* [MPM](https://httpd.apache.org/docs/2.4/en/mod/worker.html)                               - apache multi threading .

## Versioning

For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## License

This project is licensed under the LGPL License - see the [LICENSE.md](LICENSE.md) file for details

