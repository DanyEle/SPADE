from flask import Flask, render_template
from scripts.ML_PCA import *
from scripts.ML_Autoencoder import *
from scripts.Functions import *
import time
from timeloop import Timeloop
from datetime import timedelta
import matplotlib.pyplot as plt
%matplotlib inline

import requests


app = Flask(__name__)
tl = Timeloop()



##CONSTANTS OVER HERE!!

LOOP_INTERVAL_RAW_DATA=1 #seconds
LOOP_INTERVAL_TRAIN_MODEL=30 #seconds

BEAGLE_IP = "146.48.82.129"
BEAGLE_PORT = "8086"


#Data stored in this dictionary upon initializing the application
#data["timestamps"] = Format: Date + Time.The timestamp of every single data acquisition
#data["Anomaly"] = Format: Boolean. Reports whether the current data acquisition is anomalous
#data["Thresh"] = Float. Current value of the threshold identified
#data["Mob Dist"] = Float. Distance of the current data acquisition from the trained model.
data_PCA = {}



######STUFF FOR PCA MODEL######
#@app.route('/',  methods=['GET'])
#def flask_train_display_PCA_plot():
#    
#    #Then, split the data into train and test
#    #Training data: non-anomalous data (steady-state conditions)
#    #Then, given a test sample, we compute the Mahalanobis
#    #distance to the non-anomalous data
#    # and classify the test points as an “anomaly”
#    # if the distance is above a certain threshold.
#    X_train, X_test = holdout_data(merged_data)
#    
#    X_train_PCA, X_test_PCA = preprocess_data_pca(X_train, X_test)
#    
#    
#    timestamps = list(data_PCA.get("Timestamps"))
#    #Remove the useless timestamp from the string
#    timestamps_fixed = [str(s).replace('Timestamp(', '') for s in timestamps]
#    return render_template("index.html", timestamps=timestamps_fixed, distances=list(data_PCA.get("Mob dist")),
#                           anomalies=list(data_PCA.get("Anomaly")), thresholds = list(data_PCA.get("Thresh")))


#def preprocess_data_flask():
#    X_train, X_test = initialize_data()
#    
#    anomaly_all_data_PCA = train_PCA_model(X_train, X_test, matplotlib=False)
#    data_PCA["Mob dist"] = anomaly_all_data_PCA["Mob dist"]
#    data_PCA["Thresh"] = anomaly_all_data_PCA["Thresh"]
#    data_PCA["Anomaly"] = anomaly_all_data_PCA["Anomaly"]
#    data_PCA["Timestamps"] = anomaly_all_data_PCA["Timestamps"]



def main_PCA():
    merged_data = load_data_in_data_frame()

    train_display_plot_PCA_plt(merged_data)


def train_display_plot_PCA_plt(merged_data):
    #merged_data = load_data_in_data_frame()
    X_train, X_test = holdout_data(merged_data)
    #There's much more test data actually
    X_train_PCA, X_test_PCA = preprocess_data_pca(X_train, X_test)
    
    #ADVANTAGE of PCA: the threshold is identified automatically by the algorithm.
    #Assuming that the data follows a Chi^2 distribution if the assumption of normal distribution is fullfilled
    anomaly_all_data_PCA = train_PCA_model(X_train_PCA, X_test_PCA, matplotlib=True)
    plot_anomaly = anomaly_all_data_PCA.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])




def main_autoencoder_dummy():
    #data frame with indexes having the data/time and one column per axis
    merged_data = load_data_in_data_frame()
    train_display_plot_Autoencoder_plt(merged_data)



def train_display_plot_Autoencoder_plt_dummy(merged_data):
    X_train, X_test = holdout_data(merged_data)
    
    #first thing first, let's initialize the data that we will be needing to generate the autoencoder model
    autoencoder_model = create_autoencoder(X_train)
    
    trained_model, history = train_model(autoencoder_model, X_train, batch_size=10, num_epochs=100)

    #Let's see how the loss function evolved during the training process.
    show_training_history_loss_plot(history)   
    
    #And let's see where a good threshold may lie at by inspecting the loss of the training set. 
    show_loss_distr_training_set(X_train, trained_model)
    
    anomaly_threshold = find_loss_threshold_value(X_train, trained_model)
    
    test_scored_anom = mark_data_frame_as_anomaly(X_test, trained_model, threshold_value)
    
    train_scored_anom = mark_data_frame_as_anomaly(X_train, trained_model, threshold_value)
    
    scored = pd.concat([train_scored_anom, test_scored_anom])
    #Loss Mae = Mean Absolute Error. 
    scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])    
    

    
def main_autoencoder_real_BB():
    data_frame_loaded = fetch_data_from_BB(BEAGLE_IP, BEAGLE_PORT)
    
    
    

#######STUFF FOR BEAGLEBOARD#####

@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_RAW_DATA))
def run_task_regularly():
    print("Running task every " + str(LOOP_INTERVAL_RAW_DATA))
    


@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_TRAIN_MODEL))
def train_model():
    print("Training model....")





#Commands ran upon starting application
if __name__ == '__main__':
    #X_train, X_test =  initialize_data()
    tl.start(block=False)


    app.run(debug=True)


