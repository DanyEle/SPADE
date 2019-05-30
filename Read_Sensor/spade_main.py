#!/usr/bin/python3.5

import os
import sys
import time
import serial
import requests
import argparse
from influxdb import InfluxDBClient

#handle better logging
#import logging
#import logging.handlers
#import sys
#LOG_FILENAME = "/tmp/spade.log"
#logger = logging.getLogger(__name__)
#handler = logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=314572, backupCount=3)
#formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
#handler.setFormatter(formatter)
#logger.addHandler(handler)
#logger.setLevel(logging.INFO)
#
#class StreamToLogger(object):
#   """
#   Fake file-like stream object that redirects writes to a logger instance.
#   """
#   def __init__(self, logger, log_level=logging.INFO):
#      self.logger = logger
#      self.log_level = log_level
#      self.linebuf = ''
#
#   def write(self, buf):
#      for line in buf.rstrip().splitlines():
#         self.logger.log(self.log_level, line.rstrip())
#
#   def flush(self):
#       self.logger.handlers[0].flush()
#
#sys.stdout = StreamToLogger(logger, logging.INFO)
#sys.stderr = StreamToLogger(logger, logging.ERROR)
#def handle_exception(exc_type, exc_value, exc_traceback):
#    if issubclass(exc_type, KeyboardInterrupt):
#        sys.__excepthook__(exc_type, exc_value, exc_traceback)
#        return
#
#    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
#sys.excepthook = handle_exception


INFLUX_URL = "146.48.82.129"
INFLUX_PORT = 8086
INFLUX_DB_NAME = "mydb"
TABLE = "accelerometer"

def main():
	port = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=3.0)
	acc_x = 0
	acc_y = 0
	acc_z = 0
	x_avg = 0
	y_avg = 0
	z_avg = 0

	samples_number = 150
	counter = 0
	post_counter = 0

	client = InfluxDBClient(INFLUX_URL, INFLUX_PORT, database=INFLUX_DB_NAME)
	
	start_time = time.time()
	while True:
		while (counter < samples_number):
			rcv = port.read(1)
			if (rcv.hex() == '80'):
				rcv = port.read(2)
				acc_x += int.from_bytes(rcv, byteorder='big', signed=True)
				rcv = port.read(2);
				acc_y += int.from_bytes(rcv, byteorder='big', signed=True)
				rcv = port.read(2);
				acc_z += int.from_bytes(rcv, byteorder='big', signed=True)  	
				
				counter += 1;

		x_avg = acc_x / counter
		y_avg = acc_y / counter
		z_avg = acc_z / counter

		try:
			json_body = [
				{
				"measurement": TABLE,
				"fields": {
					"acc_x": float(x_avg),
					"acc_y": float(y_avg),
					"acc_z": float(z_avg),
					}
				}
			]
			print("x ", x_avg, " y ", y_avg, " z ", z_avg, "from json ", json_body[0]["fields"]["acc_x"])
			client.write_points(json_body)
			post_counter += 1

		except requests.exceptions.RequestException as e:
			print(e)
		counter = 0
		acc_x = 0 
		acc_y = 0
		acc_z = 0

		#if ((time.time() - start_time) >= 1):
		#	print("post couter ", post_counter, " each post send the averaged value over: ",samples_number, "samples \n")
		#	start_time = time.time()
		#	post_counter = 0

if __name__=="__main__":
	try:
		main()
	except:
		print("Unexpected error:", sys.exc_info(), file=sys.stderr)
