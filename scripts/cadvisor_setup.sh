sudo docker run \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --volume=/dev/disk/:/dev/disk:ro \
  --publish=8080:8080 \
  --detach=true \
  --privileged=true \
  --name=cadvisor \
  --volume=/cgroup:/cgroup:ro \
  google/cadvisor:latest


sudo /sbin/sysctl -w net.ipv4.conf.all.forwarding=1

iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
