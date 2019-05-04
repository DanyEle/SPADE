import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
import requests
import dateutil.parser
from sklearn.model_selection import train_test_split


    
def preprocess_BB_data(data_frame_loaded, shuffle=True):
    
    scaler = preprocessing.MinMaxScaler()

    X_data = pd.DataFrame(scaler.fit_transform(data_frame_loaded),
                           columns=data_frame_loaded.columns,
                           index=data_frame_loaded.index) 
    
    #Randomly shuffle the data
    if(shuffle == True):
        X_data.sample(frac=1)

    
    return(X_data)
    
    
#Input: beagleboard's IP and port where InfluxDB is running
#Output: ALL the data points the beagle board has accumulated so far
#In this function, we extract all the data points from the Beagle board
#and put all of these data points into a data frame for further processing
def get_data_frame_from_BB(beagle_ip, beagle_port):
    parameters = {}

    parameters["db"] = "mydb"
    parameters["q"] = "SELECT * FROM accelerometer" #accelerator is the DB with less data instead.

    URL = "http://" + beagle_ip +  ":" + beagle_port + "/query"
    response = requests.get(URL, params=parameters)
    response_json = response.json()
    list_results = response_json["results"]
    values_dictionary = list_results[0]
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
    

    
    