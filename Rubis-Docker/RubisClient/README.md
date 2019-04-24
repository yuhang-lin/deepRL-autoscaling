#  Containerized Rubis database

This is the Git repo of the Docker image for Rubis Client.

## Getting Started

This repository contains :

```
../RubisClient/
├── Client
│   ├── bench                               (report creation scripts)
│   │   ├── compute_global_stats.awk
│   │   ├── format_sar_output.awk
│   │   └── servlets_generate_graphs.sh
│   ├── database                            ( lists of regions and categories used to generate the database data )
│   ├── edu                                 ( Rubis Client source -  *.java )
│   ├── Makefile                            ( makefile -  make help for more infos)
│   ├── rubis.properties                    ( rubis propierties file - follow a short explanation )
│   └── workload                            ( contains the files that describes the workload ) 
├── ssh-keys                                ( ssh Keys used by the client to connect to Rubis Servlets )
│   ├── rubis_rsa_key
│   └── rubis_rsa_key.pub
├── LICENSE.md
├── README.md
└── supervisord.conf                        ( used to run and manage daemons )

```

### Important Settings

The rubis.properties file contains the most important communication settings.


httpd_hostname = *apache container name*
httpd_port = *apache port*

servlets_server = *rubis containers name*

database_server = *mysql containers name*

You can find more information about all the other settings here:

[rubis.properties](http://rubis.ow2.org/doc/properties_file.html)

NOTE: the monitoring_options has been slightly changed to support the most recent version of sar.

Determining containers name when scaling your sevice up with docker compose can be the tricky.
Usually compose assignes an host name to each container following this convention

**workingdir_servicename_instancenumber**

example: 
   
```bash
 [root@localhost]# pwd 
 /root/RubisDB

 [root@localhost]# docker-compose up --scale dbcluster=2
 Creating network "rubisdb_default" with the default driver
 Creating rubisdb_dbcluster_1 ... done
 Creating rubisdb_dbcluster_2 ... done
 Attaching to rubisdb_dbcluster_2, rubisdb_dbcluster_1
 dbcluster_2  | 2018-02-28 11:25:14,422 INFO supervisord started with pid 1
 dbcluster_1  | 2018-02-28 11:25:14,609 INFO supervisord started with pid 1
```
This would lead to database_server = rubisdb_dbcluster_2,rubisdb_dbcluster_1 in the rubis.properties.

### Build 

example :
```
docker build -t rubisclient  .
```

### RUN

This image exposes the ports 80 for Apache. This is used to publish the results of the benchmark execution.

example:

Run the container and Map TCP ports 8080 and 22 to TCP port 5000 and 2222 on the host.

example using docker cli

```
docker run -p 5000:8080 --name=client -d rubisclient
```

full example with docker-compose v2

```
version: '2'
services:
  rubis:
    image: "Rubis/rubis"
    depends_on:
     - rubisdb
  rubisdb:
    image: "Rubis/rubisdb"
  rubisweb:
    build:
      image: "Rubis/rubisweb"
    ports:
     - "5002:80"
    environment:
     - LB_MEMBER_1=dockerfiles_rubis_1
     - LB_MEMBER_2=dockerfiles_rubis_2
    depends_on:
     - rubis
  rubisclient:
    depends_on:
     - rubisdb
     - rubis
     - rubisweb
    image: "Rubis/rubisclient"
    ports:
     - "5003:80"
    environment:
     - TARGET=emulator
```

```
docker-compose up --scale rubis=2 -d rubisclient
```

This will compile the client code, fill the database tables and finally stress the system.
Once the container is up and running, go to http://HOST-IP:PORT/bench and you can see the emulation running.


#### Docker Environment Variables

-TARGET: used to specify what the client does:
         
 ```bash
               Rubis Client  
   target                         help
   ------                         ----
                               
                               
   all                            compile the client.
   initDB                         Initialize the RubisDB.
   emulator                       Start benchmarking.
   clean                          clean up all the mess removing .class.
   benchmark                      Compile the source, Init the DB and run the benchmark.
   
 ```

## Built With

* [apache](https://tomcat.apache.org/) - apache.
* [gnuplot](http://www.gnuplot.info/)  - gnuplot 4.6 patchlevel 2.
* [Rubis](http://rubis.ow2.org/doc)    - rubis client.
* [sar](http://sebastien.godard.pagesperso-orange.fr/) - sysstat version 10.1.5.
## Versioning

For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## License

This project is licensed under the LGPL License - see the [LICENSE.md](LICENSE.md) file for details

