#Use an official Centos7 runtime as a parent image
FROM centos:7

RUN yum -y update && yum clean all

#install packets required by rubis
# 1- sar;  it is used by Rubis to monitor the resource usage on a target host
# 2- tomcat8
# 3- ssh
# 4- java8
# 5- supervisord
RUN yum install -y wget \
                   openssh-server \
                   openssh-clients \
                   sysstat \
    && wget -q 'http://mirror.karneval.cz/pub/linux/fedora/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm' \
    && rpm -ivh epel-release-7-11.noarch.rpm \
    && yum install -y supervisor \
    && rm -f epel-release-7-11.noarch.rpm \
    && yum install -y \
       java-1.8.0-openjdk \
       java-1.8.0-openjdk-devel \
    && wget -q http://www-us.apache.org/dist/tomcat/tomcat-8/v8.5.40/bin/apache-tomcat-8.5.40.tar.gz \
    && tar -xvzf apache-tomcat-8.5.40.tar.gz \
    && rm -rf /apache-tomcat-8.5.40.tar.gz \
    && yum clean all \
    && rm -rf /var/cache/yum 

RUN mkdir -p /var/log/supervisor

#configure ssh daemon
RUN echo 'root:root' | chpasswd
RUN echo -e '\t StrictHostKeyChecking no'>>/etc/ssh/ssh_config \
    && echo -e '\t UserKnownHostsFile /dev/null'>>/etc/ssh/ssh_config


ENV CATALINA_HOME /apache-tomcat-8.5.40
ENV IS_DOCKER true

# install rubis Servlets
COPY servlets/rubis_servlets.war $CATALINA_HOME/webapps/

# copy configuration files for DB comm.

RUN mkdir -p /rubis_config.d
COPY servlets/mysql.properties /rubis_config.d/
COPY servlets/context.xml $CATALINA_HOME/conf/

#increase the maximum Connector URL size, this is required by Rubis in order to create new items.
RUN sed -i~ '/<Connector port="8080"/a \\t maxHttpHeaderSize="65536"' $CATALINA_HOME/conf/server.xml

#increase the maximum thread pool size (default 200), this is required by Rubis in order to stress the system.
RUN sed -i~ '/<Connector port="8080"/a \\t maxThreads="300"' $CATALINA_HOME/conf/server.xml

#increase the connectiontimeout (default 20000ms), this is required by Rubis in order to stress the system.
RUN sed -i -e 's/connectionTimeout="20000"/connectionTimeout="600000"/g' $CATALINA_HOME/conf/server.xml

# configure superviord
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN echo "files = /etc/supervisor/conf.d/*.conf" >> /etc/supervisord.conf

#copy ssh keys
RUN /usr/sbin/sshd-keygen 2>/dev/null
COPY ssh-keys/rubis_rsa_key.pub root/.ssh/authorized_keys

RUN chmod 600 /root/.ssh/authorized_keys

EXPOSE 8080

CMD ["-c","/etc/supervisord.conf"]
ENTRYPOINT ["/usr/bin/supervisord"]
