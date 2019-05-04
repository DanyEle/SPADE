from flask import Flask, render_template
from scripts.ML_PCA import *
from scripts.ML_Autoencoder import *
from scripts.Functions import *
import time
from timeloop import Timeloop
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline



import requests


app = Flask(__name__)
tl = Timeloop()


LOOP_INTERVAL_RAW_DATA=1 #seconds
LOOP_INTERVAL_TRAIN_MODEL=30 #seconds

BEAGLE_IP = "146.48.82.129"
BEAGLE_PORT = "8086"


#####GLOBAL VARIABLES FOR AUTOENCODERS#####

G_anom_threshold_AUTO = 0
G_autoenc_trained_model_AUTO = None
G_autoenc_data_frame_AUTO = pd.DataFrame()

#####GLOBAL VARIABLES FOR PCA#######

G_dist_train_PCA = []
G_anom_threshold_PCA = 0
G_inv_cov_matrix_PCA = None
G_model_PCA = None

#Data stored in this dictionary upon initializing the application
#data["timestamps"] = Format: Date + Time.The timestamp of every single data acquisition
#data["Anomaly"] = Format: Boolean. Reports whether the current data acquisition is anomalous
#data["Thresh"] = Float. Current value of the threshold identified
#data["Mob Dist"] = Float. Distance of the current data acquisition from the trained model.
#data_PCA = {}


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
#    X_train, X_test = holdout_dummy_data(merged_data)
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

#Remember for displaying with flask:
#anomaly["Timestamps"] = X_test_PCA.index


def train_PCA_dummy():
    merged_data = load_data_in_data_frame()
    X_train, X_test = holdout_dummy_data(merged_data)
    
    X_train_PCA, X_test_PCA = preprocess_data_pca(X_train, X_test)
    data_train =  np.array(X_train_PCA.values)
    
    mean_distr = train.mean(axis=0)
    #ADVANTAGE of PCA: the threshold is identified automatically by the algorithm.
    #Assuming that the data follows a Chi^2 distribution if the assumption of normal distribution is fullfilled
    anomaly_threshold_train, dist_train, inv_cov_matrix = find_anomaly_threshold_PCA(data_train, mean_distr)
    
    #Visualize the square of the Mahalanobis distance
    plot_mahab_distance_square(dist_train)
    #Visualize the mahalanobis distance itself
    plot_mahab_distance(dist_train)
    
    data_test = np.array(X_test_PCA.values)

    #Just plot the test dataset
    anomaly_test_PCA = pd.DataFrame()
    anomaly_test_PCA['Mob dist'] = dist_test #Distance of the test dataset from the training one. 
    anomaly_test_PCA['Thresh'] = anomaly_threshold_train
    
    #Assign flags, marking data as anomalous or not anomalous
    # If Mob dist above threshold: Flag as anomaly.
    anomaly_test_PCA['Anomaly'] = anomaly_test_PCA['Mob dist'] > anomaly_test_PCA['Thresh']
    anomaly_test_PCA.index = X_test_PCA.index
    
    plot_anomaly = anomaly_test_PCA.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])
  

def main_autoencoder_dummy():
    #data frame with indexes having the data/time and one column per axis
    merged_data = load_data_in_data_frame()    
    X_train, X_test = holdout_dummy_data(merged_data)
    #train_display_plot_autoencoder(X_train, X_test)
    anomaly_threshold, trained_model = autoencoder_find_anomaly_threshold(X_train)
    
    if(anomaly_threshold == False):
        print("No outliers have been identified in the sample provided")
        return
    
    print("Anomalous Threshold identified (Z-Test)" + str(anomaly_threshold))
    test_scored_anom = mark_data_frame_as_anomaly(X_test, trained_model, anomaly_threshold)
    train_scored_anom = mark_data_frame_as_anomaly(X_train, trained_model, anomaly_threshold)
    scored = pd.concat([train_scored_anom, test_scored_anom])
    #Loss Mae = Mean Absolute Error. 
    scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])    
    
    
#Input: None
#Output: the autoencoder model trained on all the data present in the BB,
#the data frame marked with anomalous values and the anomalous threshold
#are are saved in global variables, respectively
#TODO: Make this function be called every X minutes to update the model regularly. 
def train_autoencoder_BB():
    
    data_frame_train = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    X_train = preprocess_BB_data(data_frame_train) 
        
    anomaly_threshold, trained_model = autoencoder_find_anomaly_threshold(X_train)
    #Update global variable.
    anomaly_threshold = 0.2
    #if(anomaly_threshold == False):
    #    print("No outliers have been identified in the sample provided")
    #    return
    print("Anomalous Loss Threshold identified: [" + str(anomaly_threshold)+ "]")
    
    data_frame_train_anomalies = mark_data_frame_as_anomaly(X_train, trained_model, anomaly_threshold)
    
    data_frame_train_anomalies.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])   
    
    ###Finally, update the global variables:
    G_anom_threshold_AUTO = anomaly_threshold
    G_trained_model_AUTO = trained_model
    G_data_frame_AUTO = data_frame_train_anomalies
    
#Input: None
#Output: a graph displaying whether the data stored so far, concatenated
#with the latest obtained data and showing whether the data is exhibiting an
#anomalous behaviour.
def test_autoencoder_BB():
    #Example: over here, we would just need to get the data obtained every in the last 15 minutes. 
    data_frame_test = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    
    X_test = preprocess_BB_data(data_frame_loaded) 
    #use the global model and the global threshold that were last updated. 
    data_frame_test = mark_data_frame_as_anomaly(X_test, G_trained_model_AUTO, G_anom_threshold_AUTO)

    #let's concatenate the trainining dataset with the current test dataset ?
    data_frame_conc = pd.concat([G_data_frame_AUTO, data_frame_test])
    #And display all the concatenated values
    
    #Just the current data
    data_frame_test.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])   
    
    #The test data only. 
    data_frame_conc.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])   

    
def train_PCA_BB():
    data_frame_train = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    #X_train, X_test = train_test_split(X_data, test_size=0.4)
    X_train = preprocess_BB_data(data_frame_train, shuffle=True) #No need for shuffle in the test dataset.
    
    X_train_PCA, pca_model = fit_train_data_pca(data_frame_train)
    data_train =  np.array(X_train_PCA.values)
    mean_distr = data_train.mean(axis=0)
    
    #The threshold is identified automatically by the algorithm.
    #Assuming that the data follows a Chi^2 distribution if the assumption of normal distribution is fullfilled
    anomaly_threshold_train, dist_train, inv_cov_matrix = find_anomaly_threshold_PCA(data_train, mean_distr)
    
    plot_mahab_distance_square(dist_train)
    #Visualize the mahalanobis distance itself
    plot_mahab_distance(dist_train)
    
    #Save into GLOBAL VARIABLES the current state    
    G_anomaly_threshold_PCA = anomaly_threshold_train
    G_dist_train_PCA =  dist_train
    G_inv_cov_matrix_PCA = inv_cov_matrix
    G_model_PCA = pca_model
    G_mean_distr_PCA = mean_distr
    
    
    
def test_PCA_BB():
    #Example: over here, we would actually need to get the data obtained in a small time frame.
    data_frame_test = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    
    #First thing first, need to pre-process the data
    X_test = preprocess_BB_data(data_frame_test, shuffle=False)
    
    X_test_PCA = transform_test_data_PCA(X_test, pca=G_model_PCA)
    #Visualize the square of the Mahalanobis distance
    data_test = np.array(X_test_PCA.values)
    
    dist_test = MahalanobisDist(G_inv_cov_matrix_PCA, G_mean_distr_PCA, data_test, verbose=False)
    
    plot_mahab_distance_square(dist_test)
    #Visualize the mahalanobis distance itself
    plot_mahab_distance(dist_test)
    
    #Just plot the test dataset
    anomaly_test_PCA = pd.DataFrame()
    anomaly_test_PCA['Mob dist'] = dist_test #Distance of the test dataset from the training one. 
    anomaly_test_PCA['Thresh'] = anomaly_threshold_train
    
    #Assign flags, marking data as anomalous or not anomalous
    # If Mob dist above threshold: Flag as anomaly.
    anomaly_test_PCA['Anomaly'] = anomaly_test_PCA['Mob dist'] > anomaly_test_PCA['Thresh']
    anomaly_test_PCA.index = X_test_PCA.index
    
    plot_anomaly = anomaly_test_PCA.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])

    
    
        
    
    
    

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


