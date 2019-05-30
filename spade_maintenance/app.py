from scripts.ML_PCA import *
from scripts.ML_Autoencoder import *
from scripts.Functions import *
import time
from timeloop import Timeloop
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import requests


tl = Timeloop()

#Autoencoder loop interval
LOOP_INTERVAL_INFERENCE_AUTO=30 #seconds
LOOP_INTERVAL_TRAIN_MODEL_AUTO=3000 #seconds


TIME_INFERENCE_AUTO=600
TIME_TRAIN_MODEL_AUTO=120


#PCA loop interval
LOOP_INTERVAL_INFERENCE_PCA=10 #seconds
LOOP_INTERVAL_TRAIN_MODEL_PCA=120 #seconds

TIME_INFERENCE_PCA=600 #get the data from the last X seconds
TIME_TRAIN_MODEL_PCA=120 #get the data from the last X seconds


#Make sure LOOP_INTERVAL_INFERENCE_PCA > LOOP_INTERVAL_TRAIN_MODEL_PCA

#IP and port of the host where the raw data is taken from
INFLUX_IP_RAW_DATA = "146.48.82.129" #Must be string
INFLUX_PORT_RAW_DATA = "8086" #Must be string
INFLUX_TABLE_RAW_DATA = "accelerometer"

#IP and port of the host where the processed data is stored.
INFLUX_IP_PROCESSED = "146.48.82.129"
INFLUX_PORT_PROCESSED = "8086"

INFLUX_TABLE_PROCESSED_AUTO = "autoencoder" #Name of the table into which we insert the processed data from the autoencoder.
INFLUX_TABLE_PROCESSED_PCA = "PCA" #Name of the table into which we insert the processed data from the PCA 

#####GLOBAL VARIABLES FOR AUTOENCODERS#####

global G_anom_threshold_AUTO #= 0
global G_trained_model_AUTO #= None
G_trained_model_AUTO = None
global G_autoenc_data_frame_AUTO #= pd.DataFrame()

#####GLOBAL VARIABLES FOR PCA#######

global G_dist_train_PCA 
global G_anom_threshold_PCA 
global G_inv_cov_matrix_PCA 
global G_model_PCA
G_model_PCA = None
global G_mean_distr_PCA

#Input: None
#Output: the autoencoder model trained on all the data present in the BB,
#the data frame marked with anomalous values and the anomalous threshold
#are saved in global variables, respectively
def train_autoencoder_BB():
    data_frame_train = get_data_frame_from_BB(INFLUX_IP_RAW_DATA, INFLUX_PORT_RAW_DATA, INFLUX_TABLE_RAW_DATA, TIME_TRAIN_MODEL_AUTO)  
    #let's try to visualize the data got from the beagle
    #X_train, X_test = train_test_split(data_frame_train, test_size=0.2)
    #STEADY STATE
    X_train = preprocess_BB_data(data_frame_train) 
    #If training goes wrong, that means the old model is still being used!
    anomaly_threshold, trained_model, history = autoencoder_find_anomaly_threshold(X_train)
    print("Trained Autoencoder model")
    #Debug: Let's see how the loss function evolved during the training process.
    show_training_history_loss_plot(history)   
    #Debug: And let's see where a good threshold may lie at by inspecting the loss of the training set. 
    show_loss_distr_training_set(X_train, trained_model)
    #Update global variable.
    #if(anomaly_threshold == False):
    #    print("No outliers have been identified in the sample provided")
        
    print("Anomalous Loss Threshold identified: [" + str(anomaly_threshold)+ "]")
    
    data_frame_train_anomalies = mark_data_frame_as_anomaly(X_train, trained_model, anomaly_threshold)
    #Debug - plot
    data_frame_train_anomalies.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])   
    ###Finally, update the global variables 
    global G_anom_threshold_AUTO
    global G_trained_model_AUTO
    global G_data_frame_AUTO

    G_anom_threshold_AUTO = anomaly_threshold
    G_trained_model_AUTO = trained_model
    G_data_frame_AUTO = data_frame_train_anomalies

#Input: None
#Output: a graph displaying whether the data stored so far, concatenated
#with the latest obtained data and showing whether the data is exhibiting an
#anomalous behaviour.
def test_autoencoder_BB():
    #Example: over here, we would just need to get the data obtained every in the last 15 minutes. 
    data_frame_test = get_data_frame_from_BB(INFLUX_IP_RAW_DATA, INFLUX_PORT_RAW_DATA, INFLUX_TABLE_RAW_DATA, TIME_INFERENCE_AUTO) 
    
    X_test = preprocess_BB_data(data_frame_test)  #data_frame_test to get all the data
    
    #use the global model and the global threshold that were last updated. 
    data_frame_test = mark_data_frame_as_anomaly(X_test, G_trained_model_AUTO, G_anom_threshold_AUTO)

    #let's concatenate the trainining dataset with the current test dataset ?
    #Just used for display the concatenation of training + test data
    data_frame_conc = pd.concat([G_data_frame_AUTO, data_frame_test])
    #And display all the concatenated values
    
    anomaly_threshold = data_frame_conc["Anomaly"][0]
    print("Anomaly threshold in the Autoencoder inference phase [" + str(G_anom_threshold_AUTO) + " ]")
    #Debug plot - Just the current data
    data_frame_test.plot(logy=False,  figsize = (10,6),  color = ['blue','red'])   
    #new_indices = np.arange(0, len(data_frame_conc))
    #Insert the data of data_frame_test into influxDB, where it will be displayed by Grafana
    #Only insert data in the interval passed
    insert_data_frame_into_influx(data_frame_test, INFLUX_IP_PROCESSED, INFLUX_PORT_PROCESSED, INFLUX_TABLE_PROCESSED_AUTO, LOOP_INTERVAL_INFERENCE_PCA)
    print("Done inserting data!")
        
    
def train_PCA_BB():
    data_frame_train = get_data_frame_from_BB(INFLUX_IP_RAW_DATA, INFLUX_PORT_RAW_DATA, INFLUX_TABLE_RAW_DATA, TIME_TRAIN_MODEL_PCA)
    #X_train, X_test = train_test_split(data_frame_train, test_size=0.6)
    X_train_PCA = preprocess_BB_data(data_frame_train, shuffle=True)  #X_train
    X_train_PCA, pca_model = fit_train_data_pca(X_train_PCA) #X_train_PCA
    data_train =  np.array(X_train_PCA.values)
    mean_distr = data_train.mean(axis=0)
    #The threshold is identified automatically by the algorithm.
    #Assuming that the data follows a Chi^2 distribution if the assumption of normal distribution is fullfilled
    anomaly_threshold_train, dist_train, inv_cov_matrix = find_anomaly_threshold_PCA(data_train, mean_distr)
    #Visualize the mahalanobis distance computed over the training data.
    #Debug plot
    plot_mahab_distance_square(dist_train)
    #Debug plot
    plot_mahab_distance(dist_train)
    print("Trained PCA Model")
    print("Mahalanobis distance threshold identified: " + str(anomaly_threshold_train))    
    #Save into GLOBAL VARIABLES the current state   
    global G_dist_train_PCA 
    global G_anom_threshold_PCA 
    global G_inv_cov_matrix_PCA 
    global G_model_PCA
    global G_mean_distr_PCA
    
    G_anom_threshold_PCA = anomaly_threshold_train
    G_dist_train_PCA =  dist_train
    G_inv_cov_matrix_PCA = inv_cov_matrix
    G_model_PCA = pca_model
    G_mean_distr_PCA = mean_distr
    
    
def test_PCA_BB():
    #Example: over here, we would actually need to get the data obtained in a small time frame.
    data_frame_test = get_data_frame_from_BB(INFLUX_IP_RAW_DATA, INFLUX_PORT_RAW_DATA, INFLUX_TABLE_RAW_DATA, TIME_INFERENCE_AUTO)
    #First thing first, need to pre-process the data
    X_test = preprocess_BB_data(data_frame_test, shuffle=False) #X_test
    X_test_PCA = transform_test_data_PCA(X_test, pca=G_model_PCA)
    #Visualize the square of the Mahalanobis distance
    data_test = np.array(X_test_PCA.values)
    dist_test = MahalanobisDist(G_inv_cov_matrix_PCA, G_mean_distr_PCA, data_test, verbose=False)
    #Debug plot
    #plot_mahab_distance_square(dist_test)
    #Visualize the mahalanobis distance itself
    #Debug plot
    #plot_mahab_distance(dist_test)
    #Just plot the test dataset
    anomaly_test_PCA = pd.DataFrame()
    anomaly_test_PCA['Mob dist'] = dist_test #Distance of the test dataset from the training one. 
    anomaly_test_PCA['Thresh'] = G_anom_threshold_PCA
    #Assign flags, marking data as anomalous or not anomalous
    # If Mob dist above threshold: Flag as anomaly.
    anomaly_test_PCA['Anomaly'] = anomaly_test_PCA['Mob dist'] > anomaly_test_PCA['Thresh']
    anomaly_test_PCA.index = X_test_PCA.index
    #Debug plot
    plot_anomaly = anomaly_test_PCA.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])
    #anom_test_new = anomaly_test_PCA
    #Alternative non-scaled representation
    #new_indices = np.arange(0, len(anomaly_test_PCA))
    #anom_test_new.index = new_indices
    
    insert_data_frame_into_influx(anomaly_test_PCA, INFLUX_IP_PROCESSED, INFLUX_PORT_PROCESSED, INFLUX_TABLE_PROCESSED_PCA, LOOP_INTERVAL_INFERENCE_PCA)
    print("Done inserting data!")
    
    #Debug plot
    #plot_anomaly = anom_test_new.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])

###########MAIN###########

@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_TRAIN_MODEL_PCA))
def train_PCA_model_regularly():
    #Train PCA model and populate the global variables...
    print("Training PCA Model...")
    train_PCA_BB()
    
@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_INFERENCE_PCA))
def test_PCA_model_regularly():
    #If we have already trained a model, then we are able to use it for inference...
    if(G_model_PCA != None):
        print("Using PCA Model for inference...")
        test_PCA_BB()
#        
#            
#@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_TRAIN_MODEL_AUTO))
#def train_autoencoder_model_regularly():
#    print("Training Autoencoder model...")
#    train_autoencoder_BB()
#    
#@tl.job(interval=timedelta(seconds=LOOP_INTERVAL_INFERENCE_AUTO))
#def test_autoencoder_model_regularly():
#    if(G_trained_model_AUTO != None):
#        print("Using Autoencoder model for inference...")
#        test_autoencoder_BB()
    


if __name__ == "__main__":
    #Start regular jobs...
    tl.start(block=True)    


