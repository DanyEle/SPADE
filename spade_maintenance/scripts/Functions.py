import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
import requests
import re
import dateutil.parser
from sklearn.model_selection import train_test_split



def load_data_in_data_frame():
    data_dir = '/home/daniele/WNES/2nd_test'
    merged_data = pd.DataFrame()
    print("Loading the data....")
    #Use one data point every 10 minutes (i.e.: take the mean every 10 minutes)
    for filename in os.listdir(data_dir):
        print(filename)
        dataset=pd.read_csv(os.path.join(data_dir, filename), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
        dataset_mean_abs.index = [filename]
        merged_data = merged_data.append(dataset_mean_abs)

    #Daniele: 4 bearings --> 4 accelerometers' data.
    merged_data.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']

    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.sort_index()
    #merged_data.to_csv('merged_dataset_BearingTest_2.csv')
    #merged_data.head()
    return(merged_data)


def holdout_dummy_data(merged_data):
    dataset_train = merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
    dataset_test = merged_data['2004-02-13 23:52:39':]

    scaler = preprocessing.MinMaxScaler()

    X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                           columns=dataset_train.columns,
                           index=dataset_train.index)
    # Random shuffle training data
    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(dataset_test),
                          columns=dataset_test.columns,
                          index=dataset_test.index)
    
    
    return(X_train, X_test)
    
    
def preprocess_BB_data(data_frame_loaded):
    
    scaler = preprocessing.MinMaxScaler()

    X_data = pd.DataFrame(scaler.fit_transform(data_frame_loaded),
                           columns=data_frame_loaded.columns,
                           index=data_frame_loaded.index) 
    
    #Randomly shuffle the data
    X_data.sample(frac=1)

    
    return(X_data)
    
    
#Input: beagleboard's IP and port where InfluxDB is running
#Output: ALL the data points the beagle board has accumulated so far
#In this function, we extract all the data points from the Beagle board
#and put all of these data points into a data frame for further processing
def get_data_frame_from_BB(beagle_ip, beagle_port):
    parameters = {}

    parameters["db"] = "mydb"
    parameters["q"] = "SELECT * FROM accelerator"

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
    

    
    