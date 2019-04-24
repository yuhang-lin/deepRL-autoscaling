#!/bin/bash

MYHOME=$(dirname $0)
source "${MYHOME}/functions"

######################
#  CREATE CLUSTER
######################

DEPLOY_CLUSTER=( ${!CL_MEMBER_*} )
PORT_CLUSTER=33061
MYIP=$(get_ip $HOSTNAME)
ENTRYPOINT_PID=

#create password files

REP_PSWD_FILE='/var/lib/mysql-files/secret/rep'
REP_PSWD="password"
create_password_file $REP_PSWD_FILE $REP_PSWD

ROOT_PSWD_FILE='/var/lib/mysql-files/secret/root'
create_password_file $ROOT_PSWD_FILE $MYSQL_ROOT_PASSWORD


## read information from containers link - if any
if [  -z "${DEPLOY_CLUSTER}" ]; then
  DEPLOY_CLUSTER=( $(env | awk -F "=" '{print $1}' | grep '[a-zA-Z]*_[0-9]*_PORT_[0-9]*_TCP_ADDR') )
  PORT_CLUSTER=( $(env | awk -F "=" '{print $1}' | grep '[a-zA-Z]*_[0-9]*_PORT_[0-9]*_TCP_PORT') )
fi

## now loop through the above array of hostnames

if [ ! -z "${DEPLOY_CLUSTER}" ]; then

cat << EOF > /etc/mysql/my.cnf
[mysqld]
   # General replication settings
   gtid_mode = ON
   enforce_gtid_consistency = ON
   master_info_repository = TABLE
   relay_log_info_repository = TABLE
   binlog_checksum = NONE
   log_slave_updates = ON
   log_bin = binlog
   binlog_format = ROW
   transaction_write_set_extraction = XXHASH64
   loose-group_replication_bootstrap_group = OFF
   loose-group_replication_start_on_boot = OFF
   loose-group_replication_group_name = "959cf631-538c-415d-8164-ca00181be227"

   # Single or Multi-primary mode? Uncomment these two lines
   # for multi-primary mode, where any host can accept writes
   loose-group_replication_single_primary_mode = OFF
   loose-group_replication_enforce_update_everywhere_checks = ON

   server_id = ${MYIP##*.}
   bind-address = "${MYIP}"
   report_host = "${MYIP}"
   loose-group_replication_local_address = "${MYIP}:${PORT_CLUSTER}"  
EOF


for i in "${DEPLOY_CLUSTER[@]}"
do
  member=$(eval "echo \$$i")
  memberIP=$(get_ip $member)
  allIPs=${memberIP}:${PORT_CLUSTER},$allIPs
done

  #clean up, remove trailing commas.  
  allIPs=$(echo ${allIPs}| sed 's/,*$//g')
  

echo "   loose-group_replication_group_seeds = \"${allIPs}\"">>/etc/mysql/my.cnf

mastertmp=$(eval "echo \$${DEPLOY_CLUSTER[0]}")
masterIP=$(get_ip $mastertmp)

cat << EOF > /docker-entrypoint-initdb.d/100_rubis.sql
   SET SQL_LOG_BIN=0;
   CREATE USER 'replication'@'%' IDENTIFIED BY '$REP_PSWD';
   GRANT REPLICATION SLAVE ON *.* TO 'replication'@'%';
   GRANT SELECT ON performance_schema.replication_group_members TO 'replication'@'%';
   FLUSH PRIVILEGES;
   SET SQL_LOG_BIN=1;
   CHANGE MASTER TO MASTER_USER='replication', MASTER_PASSWORD='$REP_PSWD' FOR CHANNEL 'group_replication_recovery';
   INSTALL PLUGIN group_replication SONAME 'group_replication.so';
EOF

echo "starting entrypoint.sh ..... "
#run the official myslq server installation script.
nohup /entrypoint.sh mysqld --innodb-deadlock-detect=OFF  2>/var/log/entrypoint-error.log 1>/var/log/entrypoint.log &
ENTRYPOINT_PID=$!
echo "entrypont.sh started with pid $ENTRYPOINT_PID "


#wait untill myslqd is up and running ( entrypoint runs mysqld with ----skip-networking,thus preventing any TCP connection )
wait_for_mysqld $MYIP $REP_PSWD_FILE

#check if the replication plugin is correctly installed
plugin_status=$(mysql --defaults-extra-file="$ROOT_PSWD_FILE" -e "SELECT PLUGIN_STATUS \
                                      FROM information_schema.PLUGINS \
                                      WHERE PLUGIN_NAME = 'group_replication'\G"|grep ACTIVE)
check_status_and_exit_on_error
  
if [ "${plugin_status}" == "PLUGIN_STATUS: ACTIVE" ]; then
  if [ "${MYIP}" == "${masterIP}" ]; then
    mysql --defaults-extra-file="$ROOT_PSWD_FILE" -e "SET GLOBAL group_replication_bootstrap_group=ON;
                            START GROUP_REPLICATION;
                            SET GLOBAL group_replication_bootstrap_group=OFF;
                            SET GLOBAL auto_increment_offset = 1;
                            SET GLOBAL auto_increment_increment = 1;"
    check_status_and_exit_on_error
  else
    #check if other members are online
    echo "check availability of cluster members for MEMBER: $MYIP"
    for i in "${DEPLOY_CLUSTER[@]}"
    do
       member=$(eval "echo \$$i")
       memberIP=$(get_ip $member)

       if [ "${memberIP}" == "${masterIP}" ]; then
         wait_for_master $masterIP $REP_PSWD_FILE
       else
         wait_for_mysqld $memberIP $REP_PSWD_FILE
       fi
    done
    echo "MEMBER: $MYIP - all other members are up and running, JOIN the cluster"
    mysql --defaults-extra-file="$ROOT_PSWD_FILE" -e "START GROUP_REPLICATION;"
    check_status_and_exit_on_error
   
    # server specific configuration - replication
    # this setting guarantees that auto-increment values are assigned in a predictable and repeatable order
    # it turns out that Rubis needs the IDs in items being consecutive and equal to the item ID.

    mysql --defaults-extra-file="$ROOT_PSWD_FILE" -e "SET GLOBAL auto_increment_offset = 1;
                                                      SET GLOBAL auto_increment_increment = 1;"
    check_status_and_exit_on_error
  fi
else
  >&2 echo "Undefined plugin status -> ${plugin_status}"
  exit 1
fi

  unset REP_PSWD

  wait $ENTRYPOINT_PID

  check_status_and_exit_on_error

else
  echo "cluster is not defined"
  echo "starting entrypoint.sh ..... "

  #run the official myslq server installation script.
  /entrypoint.sh mysqld
  
fi

