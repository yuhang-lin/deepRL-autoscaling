FROM centos:7.4.1708

RUN yum -y update
RUN yum install -y wget \
                   httpd \
                   openssh-server \
                   openssh-clients \
                   sysstat

# install epel repo
RUN wget 'http://mirror.karneval.cz/pub/linux/fedora/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm' \
    && rpm -ivh epel-release-7-11.noarch.rpm \
    && yum clean all \
    && rm -rf /var/cache/yum

#install supervisord

RUN yum install -y --enablerepo=epel \
                   supervisor \
                   python-pip \
    && yum clean all \
    && rm -rf /var/cache/yum             

# install the python docker client
RUN pip install docker

#configure supervisord

RUN mkdir -p /var/log/supervisor
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN echo "files = /etc/supervisor/conf.d/*.conf" >> /etc/supervisord.conf

#create ssh keys

RUN /usr/sbin/sshd-keygen 2>/dev/null

COPY ssh-keys/rubis_rsa_key.pub root/.ssh/authorized_keys

RUN chmod 600 /root/.ssh/authorized_keys

ENV WEB_NAME web
#ENV MEMBER_1 rubis

COPY ./cluster /root/cluster
RUN chmod +x root/.ssh/authorized_keys

#configure apache2.4 (enable monitoring and set a OVERSIZED thread pool)
RUN rm -f /etc/httpd/conf.modules.d/00-mpm.conf
COPY ./conf/11-mod_status.conf /etc/httpd/conf.modules.d/
COPY ./conf/00-mpm.conf /etc/httpd/conf.modules.d/

RUN sed -i '/#ServerName www.example.com:80/a ServerName localhost' /etc/httpd/conf/httpd.conf

RUN rm -f /var/www/html/index.html 

# forward access and error logs to docker log collector
#RUN ln -sf /dev/stdout /var/log/httpd/access_log \
#	&& ln -sf /dev/stderr /var/log/httpd/error_log

EXPOSE 80

CMD ["-c","/etc/supervisord.conf"]
ENTRYPOINT ["/usr/bin/supervisord"]
