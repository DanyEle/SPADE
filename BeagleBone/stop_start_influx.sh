sudo service influxdb stop

screen -d -m sudo influxd run -config /etc/influxdb/influxdb.conf 
