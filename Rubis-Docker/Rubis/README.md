#  Containerized Rubis Servlets

This is the Git repo of the Docker image for Rubis Servlets.

## Getting Started

This repository contains :

```
.
├── servlets                        (Rubis war file and related configurations)
│   ├── context.xml                 (Tomcat DB Connection Pool - referr to this file if you want to set the DB connection properly)
│   ├── mysql.properties            (Rubis DB configurations - OUTDATED, this Rubis version uses the Tomcat DB pool.)
│   └── rubis_servlets.war          (war file containing Rubis Servlets)
├── ssh-keys                        (ssh Keys used by the client to connect to Rubis Servlets)
│   ├── rubis_rsa_key
│   └── rubis_rsa_key.pub
├── supervisord.conf                (used to run and manage daemons)
```

### Important Settings

This version of Rubis relies on the database connection pool implementation in Apache Tomcat. All the relevant configurations can be found in context.xml.
Bear in mind that, before to BUILD/RUN this image, at least a running instance of the RubisDB is required and its name has to be provided in the context.xml.
With a Multy-Primary mysql mode, the Connector/J can be used in ``load balance`` or ``failover`` mode to forward connections to DB replicas.

example of failover with a Multi-Primary database configuration
```
<Resource name="jdbc/Rubismysql" auth="Container" type="javax.sql.DataSource"
               maxTotal="100" maxIdle="30" maxWaitMillis="30000" validationQuery="/* ping */"
               username="rubis" password="rubis" driverClassName="com.mysql.jdbc.ReplicationDriver"
               url="jdbc:mysql://dockerfiles_rubisdb_1:3306,dockerfiles_rubisdb_2:3306)/rubis"
/>
```
See the context.xml for more detailed configuration examples.

### Build 

example :
```
docker build -t rubisservlets  .
```

### RUN

This image exposes the ports 8080 and 22 for respectively tomcat and ssh.

example using docker cli:

Run the container and Map TCP ports 8080 and 22 to TCP port 5000 and 2222 on the host.

```
docker run -p 5000:8080 -p 2222:22 --name=rubis -d rubisservlets
```

example using docker-compose.yml V>=2

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
    environment:
     - CL_MEMBER_1=dockerfiles_rubisdb_1
     - CL_MEMBER_2=dockerfiles_rubisdb_2
```

See the rubisdb dscription to get more insight about DBs configuration.

## Built With

* [Tomcat](https://tomcat.apache.org/download-80.cgi) - Tomcat.
* [Java8](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) - JavaSE.
* [RubisServlet](https://docs.docker.com/docker-hub/official_repos/) - Rubis Servlets.
* [mysql Connector/J](https://dev.mysql.com/doc/connector-j/5.1/en/connector-j-usagenotes-j2ee-concepts-managing-load-balanced-connections.html) - MySQL ConnectorJ.

## Versioning

For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## License

This project is licensed under the LGPL License - see the [LICENSE.md](LICENSE.md) file for details

