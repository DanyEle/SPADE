import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
import requests
import dateutil.parser
from sklearn.model_selection import train_test_split

import time
import datetime

INXFLUX_HOST = "146.48.82.95"
INFLUX_PORT = 8086

INFLUX_INSERT_HOST = "146.48.82.129"
INFLUX_INSER_PORT = 8086

    
def preprocess_BB_data(data_frame_loaded, shuffle=True):
    scaler = preprocessing.MinMaxScaler()

    X_data = pd.DataFrame(scaler.fit_transform(data_frame_loaded),
                           columns=data_frame_loaded.columns,
                           index=data_frame_loaded.index) 
    #Randomly shuffle the data
    if(shuffle == True):
        X_data.sample(frac=1)

    return(X_data)
    
#Input: beagle IP: beagleboard's IP of InfluxDB is running
#       beagle port: beagleboard's PORT of where InfluxDB is running
#       seconds: if == -1, then get all the data from the table
#                if >= 1, then get the data from the last 'seconds' from the table
#Output: ALL the data points the beagle board has accumulated so far
#In this function, we extract all the data points from the Beagle board
#and put all of these data points into a data frame for further processing
def get_data_frame_from_BB(beagle_ip, beagle_port, influx_table, seconds=-1):
    parameters = {}
    parameters["db"] = "mydb"
    #Need to get all data in the past X minutes(?)
    
    #Get all the data
    if(seconds == -1):
        parameters["q"] = "SELECT * FROM " + str(influx_table) 
    elif(seconds >= 1):
        parameters["q"] = "SELECT * FROM " + str(influx_table) + " WHERE time <= now() AND time >= now() - " + str(seconds) + "s"

    URL = "http://" + beagle_ip +  ":" + str(beagle_port) + "/query"
    response = requests.get(URL, params=parameters)
    response_json = response.json()
    list_results = response_json["results"]
    values_dictionary = list_results[0]
    #check if values_dictionary has key "series"
    
    if 'series' not in values_dictionary:
        print("No data points fetched from the BB")
        return(None)
        
    values_series = values_dictionary["series"]
    series_dictionary = values_series[0]
    list_data_points = series_dictionary["values"]
    
    data_frame = pd.DataFrame(list_data_points, columns=["Timestamp", "X", "Y", "Z"])
    
    column_ts = parse_timestamps_column(data_frame["Timestamp"] )
    #Notice that now that the timestamps go at the index!
    data_frame.index = column_ts
    data_frame.drop(columns=["Timestamp"], inplace=True)
    del data_frame.index.name
    
    print("Fetched " + str(len(data_frame["X"])) + " data points from the BB")
    
    return(data_frame)
    
   
 
#Input: a Series of timestamps in ISO-8601 format, like:
#2019-05-01T16:01:19.168352264Z
#Output: A series of parsed timestamps 
def parse_timestamps_column(column_ts):
    column_ts_parsed = column_ts.apply(lambda ts : dateutil.parser.parse(ts))       
    pd_column_ts = pd.to_datetime(column_ts_parsed)
    return(pd_column_ts)
    

def insert_data_frame_into_influx(data_frame_test, influx_ip, influx_port, table_name):
    print("Inserting data into table [" + str(table_name) + "] at " + str(influx_ip) + ":" + str(influx_port) + "...")
    for i in range(len(data_frame_test)):
        #print(acc_x[i],acc_y[i],acc_z[i])
        #row is a list!
        row = data_frame_test.iloc[i]
        #loss = row[0]
        #threshold=row[1]
        #anomaly=row[2]r
        #print(row)
        timestamp= row.name.value #Unix formatting
        #Access the different row values with increasing indices
        #Apart from the timestamp, which is the row's index --> Needs to be access via ".name"
        command = """curl -d "{} distance={},threshold={},anomaly={} {}" -X POST http://{}:{}/write?db=mydb""".format(str(table_name), str(row[0]), str(row[1]), str(row[2]), str(timestamp), str(influx_ip), str(influx_port))
        #print(command)
        os.system(command)
        
        
#X_data can be either: X_train or X_test

    
    
    