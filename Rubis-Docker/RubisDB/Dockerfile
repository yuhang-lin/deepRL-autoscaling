#pull the my sql docker container

FROM mysql/mysql-server:5.7

ENV MYSQL_DATABASE rubis
ENV MYSQL_USER rubis
ENV MYSQL_PASSWORD rubis
ENV MYSQL_ROOT_PASSWORD root

EXPOSE 3306

#clustering 
EXPOSE 33061

# copy sql conf files (db, user, tab creation)

COPY ./database/*.sql /docker-entrypoint-initdb.d/

#install and run supervisord, ssh, scp, sar

RUN yum install -y wget \
                   sysstat \
                   openssh-server \
                   openssh-clients \
                   bind-utils \
    && wget 'http://mirror.karneval.cz/pub/linux/fedora/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm' \
    && rpm -ivh epel-release-7-11.noarch.rpm \
    && yum install -y supervisor \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN mkdir -p /var/log/supervisor

COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN echo "files = /etc/supervisor/conf.d/*.conf" >> /etc/supervisord.conf

#configuring ssh and keys creation

RUN echo 'root:root' | chpasswd
RUN echo -e '\t StrictHostKeyChecking no'>>/etc/ssh/ssh_config \
    && echo -e '\t UserKnownHostsFile /dev/null'>>/etc/ssh/ssh_config
RUN /usr/sbin/sshd-keygen 2>/dev/null
COPY ssh-keys/rubis_rsa_key.pub root/.ssh/authorized_keys

RUN chmod 600 /root/.ssh/authorized_keys

#cluster settings
RUN mkdir -p /etc/mysql/conf.d
COPY ./cluster /root/cluster

CMD ["-c","/etc/supervisord.conf"]
ENTRYPOINT ["/usr/bin/supervisord"]

#overwrite mysql image healthcheck

HEALTHCHECK CMD /healthcheck.sh
