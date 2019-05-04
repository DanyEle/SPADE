# -*- coding: utf-8 -*-



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
    
  

