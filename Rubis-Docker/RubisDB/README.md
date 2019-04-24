#  Containerized Rubis database

This is the Git repo of the Docker image for Rubid database wich uses mysql.
This image extends the mysql official image. 

## Getting Started

This repository contains :

```
RubisDB
├── cluster               ( scripts used to configure MySQL Group Replication 
│   ├── functions           with a Multi-Primary environment )    
│   └── set_up_cluster.sh 
├── database              ( sql scripts to initialize the Rubis DB 
│   ├── 10_rubis.sql        modify those scripts in order to customize
│   ├── 20_regions.sql      the Rubis database                     )
│   └── 30_categories.sql
├── docker-compose.yml    ( example of Multi_primary environment )
├── Dockerfile 
├── ssh-keys              ( used by RubisClient to connect to the DB instance )
│   ├── rubis_rsa_key
│   └── rubis_rsa_key.pub
└── supervisord.conf      ( used to start and manage the daemon processes )
```

### Build 

example :
```
docker build -t rubisdatabase  .
```

### RUN ( single mode )

This image exposes the ports 3306 and 22 for the respectively mysql and ssh daemon.

example:

Run the container and Map TCP ports 3306 and 22 to TCP port 5006 and 2225 on the host.

```
docker run -p 5006:3306 -p 2225:22 --name=rubisdb -d rubisdatabase
```
### RUN ( Multi-Primary mode )

In multi-primary mode every container is able to execute read/write transactions, and all replicas have a consistent view of the database.

![picture alt](https://dev.mysql.com/doc/refman/5.7/en/images/multi-primary.png)

Refer to the link below to get more insight.

To run the DB in a multi-primary mode you need to specify replicas hostnames as enviroment variables in each container :

     CL_MEMBER_1=container_name_1
     CL_MEMBER_2=container_name_2
     ....
     CL_MEMBER_N=container_name_N

Container name and environment variables can be:
  - set using --name option or the one provided by docker compose.
    example :
    ```
     docker run --name=mysql1 -e CL_MEMBER_1='mysql1' -e CL_MEMBER_2='mysql2' -d rubisdatabase
     docker run --name=mysql2 -e CL_MEMBER_1='mysql1' -e CL_MEMBER_2='mysql2' -d rubisdatabase
    ```
  - set featuring the name automatically assigned by docker-compose, do not assign a name with container_name: tag.
    
    See docker-compose.yml for an example. The number of replicas/containers can be scale up with the --scale option:
    ```
    docker-compose up --scale dbcluster=3

    ```
To verify the Cluster Status :

   - log in a container 
     ```
     docker exec -it rubisdb_dbcluster_1 bash
     ```
   - run the following query
     ```ruby
     bash-4.2# mysql -proot                  

     mysql> SELECT * FROM performance_schema.replication_group_members;
     +---------------------------+--------------------------------------+-------------+-------------+--------------+
     | CHANNEL_NAME              | MEMBER_ID                            | MEMBER_HOST | MEMBER_PORT | MEMBER_STATE |
     +---------------------------+--------------------------------------+-------------+-------------+--------------+
     | group_replication_applier | e3a54a77-1bd4-11e8-a9c9-0242ac120002 | 172.18.0.2  |        3306 | ONLINE       |
     | group_replication_applier | e3c825b8-1bd4-11e8-ab2a-0242ac120003 | 172.18.0.3  |        3306 | ONLINE       |
     | group_replication_applier | e3caabfc-1bd4-11e8-aa04-0242ac120004 | 172.18.0.4  |        3306 | ONLINE       |
     +---------------------------+--------------------------------------+-------------+-------------+--------------+
     3 rows in set (0.00 sec)
     ```
    

## Built With

* [mysql](https://hub.docker.com/r/mysql/mysql-server/)                                     - official mysql server image.
* [rubis](http://rubis.ow2.org/)                                                            - Rubis.
* [mysql group replication](https://dev.mysql.com/doc/refman/5.7/en/group-replication.html) - MySQL Group Replication.

## Limitations

The Multy-Primary mode can be usefull to simulate a production environment where databases in a cluster share DDL/DML Operations.
In such scenario, updated tables rows have to be actualised to achieve Consistency. 
This generates load on computing resources, contributing to further stress the system.
Unfortunately, Rubis does not offer the opportunity to load-balance writes between the nodes of the cluster, since concurrent transactions
on same table's rows cause deadlocks, see mySQL Group Replication Limitations.
[READ MORE](https://dev.mysql.com/doc/refman/5.7/en/group-replication-limitations.html)

This does not exclude the possibility to use the Multy-primary and use the ConnectorJ with a failover configuration. This allows anyway to further stress the system, 
as computing load is generate to bring secondary servers up to date in a replication scenario.

Or to further develope Rubis -- who knows!

## Versioning

For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## License

This project is licensed under the LGPL License - see the [LICENSE.md](LICENSE.md) file for details

