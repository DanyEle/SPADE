from scripts.ML_PCA import *
from scripts.ML_Autoencoder import *
from scripts.Functions import *
import time
from timeloop import Timeloop
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.model_selection import train_test_split
import requests


LOOP_INTERVAL_RAW_DATA=1 #seconds
LOOP_INTERVAL_TRAIN_MODEL=30 #seconds

INFLUX_IP = "146.48.82.129"
INFLUX_PORT = "8086"


#####GLOBAL VARIABLES FOR AUTOENCODERS#####

G_anom_threshold_AUTO = 0
G_autoenc_trained_model_AUTO = None
G_autoenc_data_frame_AUTO = pd.DataFrame()

#####GLOBAL VARIABLES FOR PCA#######

G_dist_train_PCA = []
G_anom_threshold_PCA = 0
G_inv_cov_matrix_PCA = None
G_model_PCA = None
G_mean_distr_PCA = []


  
#Input: None
#Output: the autoencoder model trained on all the data present in the BB,
#the data frame marked with anomalous values and the anomalous threshold
#are are saved in global variables, respectively
#TODO: Make this function be called every X minutes to update the model regularly. 
def train_autoencoder_BB():
    
    data_frame_train = get_data_frame_from_BB(INFLUX_IP, INFLUX_PORT)  
    #let's try to visualize the data got from the beagle
    #X_train, X_test = train_test_split(data_frame_train, test_size=0.2)
    
    #STEADY STATE
    X_train = data_frame_train[0:4000]
    
    #ANOMALOUS 
    X_test = data_frame_train[4001:4500]
    
    X_train = preprocess_BB_data(X_train) #data_frame_train 
        
    #If training goes wrong, that means the old model is still being used!
    anomaly_threshold, trained_model = autoencoder_find_anomaly_threshold(X_train)
    #Update global variable.
    #anomaly_threshold = 0.2
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
    
    X_test = preprocess_BB_data(X_test)  #data_frame_test to get all the data
    
    #use the global model and the global threshold that were last updated. 
    data_frame_test = mark_data_frame_as_anomaly(X_test, G_trained_model_AUTO, G_anom_threshold_AUTO)

    #let's concatenate the trainining dataset with the current test dataset ?
    data_frame_conc = pd.concat([G_data_frame_AUTO, data_frame_test])
    #And display all the concatenated values
    
    #Just the current data
    data_frame_test.plot(logy=False,  figsize = (10,6),  color = ['blue','red'])   
    
    new_indices = np.arange(0, len(data_frame_conc))
    data_frame_new_indexes = data_frame_conc
    data_frame_new_indexes.index = new_indices
    #Current data and the test data
    data_frame_new_indexes.plot(logy=False,  figsize = (10,6), color = ['blue','red'])   

    #TODO: insert the data of data_frame_test into influxDB, where it will displayed.
    insert_data_frame_into_influx(data_frame_test)
        
    
    
def train_PCA_BB():
    data_frame_train = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    
    X_train, X_test = train_test_split(data_frame_train, test_size=0.6)
    
    X_train_PCA = preprocess_BB_data(X_train, shuffle=True)  #data_frame_train
    
    X_train_PCA, pca_model = fit_train_data_pca(X_train_PCA) #X_train_PCA
    data_train =  np.array(X_train_PCA.values)
    mean_distr = data_train.mean(axis=0)
    #The threshold is identified automatically by the algorithm.
    #Assuming that the data follows a Chi^2 distribution if the assumption of normal distribution is fullfilled
    anomaly_threshold_train, dist_train, inv_cov_matrix = find_anomaly_threshold_PCA(data_train, mean_distr)
    
    plot_mahab_distance_square(dist_train)
    #Visualize the mahalanobis distance itself
    plot_mahab_distance(dist_train)
    
    print("Mahalanobis distance threshold identified: " + str(anomaly_threshold_train))
    
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
    X_test = preprocess_BB_data(X_test, shuffle=False) #data_frame_test
    
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

    anom_test_new = anomaly_test_PCA
    #Alternative non-scaled representation
    new_indices = np.arange(0, len(anomaly_test_PCA))
    anom_test_new.index = new_indices
    
    plot_anomaly = anom_test_new.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])

    

#######STUFF FOR BEAGLEBOARD#####

@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_RAW_DATA))
def run_task_regularly():
    print("Running task every " + str(LOOP_INTERVAL_RAW_DATA))
    


@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_TRAIN_MODEL))
def train_model():
    print("Training model....")

tl = Timeloop()
tl.start(block=False)

