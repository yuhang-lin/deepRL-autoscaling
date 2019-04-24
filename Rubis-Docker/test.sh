#/usr/bin/sh

if [ $# -ne 1 ]; then
	echo "Usage: ./test <client-target> (e.g. emulator, benchmark)"
	exit 1
fi

#start rubis database.
echo "starting rubis database"
docker-compose -p rubis up -d --no-deps --build --force-recreate rubisdb
rubis_db_cname='rubis_rubisdb_1'
rubis_db_cid=`docker ps -aqf name=$rubis_db_cname`
NL='
'
rubis_db_cid=${rubis_db_cid%%"$NL"*}
echo "rubis database started with container id $rubis_db_cid"

#now add the container name into /etc/hosts file. Then start
#servlets server.
echo "waiting for 5 secs."
sleep 5
cp $PWD/Rubis/servlets/context.xml.backup $PWD/Rubis/servlets/context.xml
echo "created base context.xml"
sed -i "s|rubis_db_cid|$rubis_db_cid|g" $PWD/Rubis/servlets/context.xml
echo "updated Rubis/servlets/context.xml with rubis database container id"
echo "building rubis servlets server"
docker-compose -p rubis up -d --no-deps --build --force-recreate rubis
rubis_server_cname='rubis_rubis_1'
rubis_server_cid=`docker ps -aqf name=$rubis_server_cname`
echo "rubis servlets server started with container id $rubis_server_cid"

#start webserver.
echo "waiting for 5 secs."
sleep 5
echo "starting rubis webserver"
docker-compose -p rubis up -d --no-deps --build --force-recreate rubisweb
rubis_webserver_cname='rubis_rubisweb_1'
rubis_webserver_cid=`docker ps -aqf name=$rubis_webserver_cname`
echo "rubis webserver started with container id $rubis_webserver_cid"


#start rubisclient
echo "waiting for 10 secs."
sleep 10
cp $PWD/RubisClient/Client/rubis.properties.backup $PWD/RubisClient/Client/rubis.properties
echo "created base rubis.properties"
sed -i "s|database_server\ =\ database_server|database_server\ =\ $rubis_db_cid|g" $PWD/RubisClient/Client/rubis.properties
rubis_db_cid=`docker ps -aqf name=$rubis_webserver_cname`
rubis_db_cid=${rubis_db_cid%%"$NL"*}
echo "updated RubisClient/Client/rubis.properties with db server"
sed -i "s|httpd_hostname\ =\ httpd_hostname|httpd_hostname\ =\ $rubis_db_cid|g" $PWD/RubisClient/Client/rubis.properties
echo "updated RubisClient/Client/rubis.properties with httpd webserver"
rubis_db_cid=`docker ps -aqf name=$rubis_server_cname`
rubis_db_cid=${rubis_db_cid%%"$NL"*}
sed -i "s|servlets_server\ =\ servlets_server|servlets_server\ =\ $rubis_db_cid|g" $PWD/RubisClient/Client/rubis.properties
echo "updated RubisClient/Client/rubis.properties with servlets server"
cp $PWD/docker-compose.yml.backup $PWD/docker-compose.yml
echo "created base docker-compose.yml"
sed -i "s|TARGET=|TARGET=$1|g" $PWD/docker-compose.yml
echo "updated docker-compose.yml with TARGET=$1"
echo "starting rubis client with $1 target"
docker-compose -p rubis up -d --no-deps --build --force-recreate rubisclient

