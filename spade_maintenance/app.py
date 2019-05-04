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


#X_data_PCA = X_train_PCA or X_test_PCA
def find_anomaly_threshold_PCA(data_train, mean_distr):
    #compute covariance matrix
    covariance_matrix, inv_cov_matrix = cov_matrix(data_train)
    #calculate the Mahalanobis distance for the training data defining “normal conditions”,
    #and find the threshold value to flag datapoints as an anomaly.
    dist_data = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    anomaly_threshold = MD_threshold(dist_data, extreme=True)
    
    return anomaly_threshold, dist_data, inv_cov_matrix



def main_PCA():
    merged_data = load_data_in_data_frame()

    train_display_plot_PCA_plt(merged_data)


def train_PCA_dummy(merged_data):
    #merged_data = load_data_in_data_frame()
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


def autoencoder_find_anomaly_threshold(X_data):
    #first thing first, let's initialize the data that we will be needing to generate the autoencoder model
    autoencoder_model = create_autoencoder(X_data)
    trained_model, history = train_model(autoencoder_model, X_data, batch_size=1, num_epochs=100)
    #Let's see how the loss function evolved during the training process.
    show_training_history_loss_plot(history)   
    #And let's see where a good threshold may lie at by inspecting the loss of the training set. 
    show_loss_distr_training_set(X_data, trained_model)
    anomaly_threshold = find_loss_threshold_value(X_data, trained_model)
    
    return anomaly_threshold, trained_model  
  

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
#Output: the trained model based on all the data present in the BB.
def train_autoencoder_BB():
    
    data_frame_loaded = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    
    X_data = preprocess_BB_data(data_frame_loaded) 
        
    anomaly_threshold, trained_model = autoencoder_find_anomaly_threshold(X_train)
    
    anomaly_threshold = 0.2
    #Let's mark the data points 
    #if(anomaly_threshold == False):
    #    print("No outliers have been identified in the sample provided")
    #    return
    print("Anomalous Threshold identified (Z-Test)" + str(anomaly_threshold))
    
    data_scored_anom = mark_data_frame_as_anomaly(X_data, trained_model, anomaly_threshold)
    
    data_scored_anom.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])    

    
    
def train_PCA_BB():
    
    data_frame_loaded = get_data_frame_from_BB(BEAGLE_IP, BEAGLE_PORT)
    
    X_data = preprocess_BB_data(data_frame_loaded) 
    
    X_train, X_test = train_test_split(X_data, test_size=0.4)
       
    X_train_PCA, X_test_PCA = preprocess_data_pca(X_train, X_test)
    data_train =  np.array(X_train_PCA.values)
    
    mean_distr = data_train.mean(axis=0)
    #The threshold is identified automatically by the algorithm.
    #Assuming that the data follows a Chi^2 distribution if the assumption of normal distribution is fullfilled
    anomaly_threshold_train, dist_train, inv_cov_matrix = find_anomaly_threshold_PCA(data_train, mean_distr)
    
    #Visualize the square of the Mahalanobis distance
    plot_mahab_distance_square(dist_train)
    #Visualize the mahalanobis distance itself
    plot_mahab_distance(dist_train)
    
    
    data_test = np.array(X_test_PCA.values)
    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
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


