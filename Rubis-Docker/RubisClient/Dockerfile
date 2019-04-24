FROM centos:7

# install all required packages
RUN yum -y update \ 
    && yum install -y httpd \
                      wget \
                      make \
                      gnuplot \ 
                      openssh-server \
                      openssh-clients \
                      sysstat \
    && wget -q 'http://mirror.karneval.cz/pub/linux/fedora/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm' \
    && rpm -ivh epel-release-7-11.noarch.rpm \
    && yum install -y supervisor \
    && yum install -y \
       java-1.8.0-openjdk \
       java-1.8.0-openjdk-devel \
    && yum clean all \
    && rm -rf /var/cache/yum


ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-1.8.0.212.b04-0.el7_6.x86_64
ENV PATH $PATH:$JAVA_HOME/bin

#configure supervisord

RUN mkdir -p /var/log/supervisor
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN echo "files = /etc/supervisor/conf.d/*.conf" >> /etc/supervisord.conf

#create ssh keys

RUN /usr/sbin/sshd-keygen 2>/dev/null
COPY ./ssh-keys/rubis_rsa_key /root/.ssh/id_rsa
COPY ./ssh-keys/rubis_rsa_key.pub root/.ssh/authorized_keys
RUN echo -e '\tStrictHostKeyChecking no'>>/etc/ssh/ssh_config \
    && echo -e '\tUserKnownHostsFile /dev/null'>>/etc/ssh/ssh_config

RUN chmod 600 /root/.ssh/id_rsa

RUN chmod 600 /root/.ssh/authorized_keys

#Rubis Client installation
COPY Client /var/www/Client

RUN rm -f /var/www/html/index.html \ 
    && ln -s /var/www/Client/bench /var/www/html/bench

ENV RUBIS_HOME /var/www/Client
ENV TARGET benchmark

EXPOSE 80

CMD ["-c","/etc/supervisord.conf"]
ENTRYPOINT ["/usr/bin/supervisord"]
