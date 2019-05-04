# -*- coding: utf-8 -*-



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

