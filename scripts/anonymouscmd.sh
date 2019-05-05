#written: Srinivas Parasa
#Usage: useful commands for the projects

#command to enable ip forwarding
sudo /sbin/sysctl -w net.ipv4.conf.all.forwarding=1

#install net tools
sudo yum install net-tools

#commands to generate and visulaise key
ssh-keygen -t rsa -b 4096 -C “sparasa@ncsu.edu”
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
vi ~/.ssh/id_rsa.pub


#install git
yes|sudo yum install git
