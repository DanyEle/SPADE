from flask import Flask, render_template
from scripts.ML_PCA import *
from scripts.Functions import *
import time
from timeloop import Timeloop
from datetime import timedelta

import requests


app = Flask(__name__)
tl = Timeloop()



##CONSTANTS OVER HERE!!

LOOP_INTERVAL_RAW_DATA=1 #seconds
LOOP_INTERVAL_TRAIN_MODEL=30 #seconds

BEAGLE_IP = "146.48.82.129"
PORT = "8086"


#Data stored in this dictionary upon initializing the application
#data["timestamps"] = Format: Date + Time.The timestamp of every single data acquisition
#data["Anomaly"] = Format: Boolean. Reports whether the current data acquisition is anomalous
#data["Thresh"] = Float. Current value of the threshold identified
#data["Mob Dist"] = Float. Distance of the current data acquisition from the trained model.
data_PCA = {}


@app.route('/',  methods=['GET'])
def show_plot_PCA():
    timestamps = list(data_PCA.get("Timestamps"))
    #Remove the useless timestamp from the string
    timestamps_fixed = [str(s).replace('Timestamp(', '') for s in timestamps]
    return render_template("index.html", timestamps=timestamps_fixed, distances=list(data_PCA.get("Mob dist")),
                           anomalies=list(data_PCA.get("Anomaly")), thresholds = list(data_PCA.get("Thresh")))


def initialize_data():
    #Firstly, load the dataset
    merged_data = load_data_in_data_frame()

    #Then, split the data into train and test
    #Training data: non-anomalous data (steady-state conditions)
    #Then, given a test sample, we compute the Mahalanobis
    #distance to the non-anomalous data
    # and classify the test points as an “anomaly”
    # if the distance is above a certain threshold.
    X_train, X_test = holdout_data(merged_data)

    return(X_train, X_test)

    #In our case, we will use a pre-acquired sample of steady-state data
    #to train the model and newly acquired test data to check
    #if the current state is anomalous.


    #print("Model successfully created")
    #plot_anomaly = anomaly_all_data.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])


@app.route('/raw_data',  methods=['GET'])
def get_raw_data_beagle():
    #need to fetch data from the Beagleboard over here.


    return 0



@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_RAW_DATA))
def fetch_raw_data():
    print("Performing request to beagle board")

    parameters = {}

    parameters["db"] = "mydb"
    parameters["q"] = "SELECT * FROM accelerator"

    URL = "http://" + BEAGLE_IP +  ":" + PORT + "/query"

    response = requests.get(URL, params=parameters)
    response_json = response.json()

    list_results = response_json["results"]

    values_dictionary = list_results[0]

    values_series = values_dictionary["series"]

    series_dictionary = values_series[0]

    list_of_lists = series_dictionary["values"]

    #Ogni singola lista è una triple di valori dell'accelerometro (X,Y,Z).

    print(list_of_lists)


@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_TRAIN_MODEL))
def train_model():
    print("Training model....")





#Commands ran upon starting application
if __name__ == '__main__':
    #X_train, X_test =  initialize_data()
    #anomaly_all_data_PCA = train_PCA_model(X_train, X_test)

    #return anomaly_all_data
    #anomaly_all_data_PCA = initialize_data()
    #data_PCA["Mob dist"] = anomaly_all_data_PCA["Mob dist"]
    #data_PCA["Thresh"] = anomaly_all_data_PCA["Thresh"]
    #data_PCA["Anomaly"] = anomaly_all_data_PCA["Anomaly"]
    #data_PCA["Timestamps"] = anomaly_all_data_PCA["Timestamps"]
    tl.start(block=False)


    app.run(debug=True)


