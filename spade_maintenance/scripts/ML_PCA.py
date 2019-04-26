# Common imports
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#%matplotlib inline

########Initialize function##############

def load_data_in_data_frame():
    data_dir = '/home/daniele/WNES/2nd_test'
    merged_data = pd.DataFrame()
    print("Loading the data....")
    #Use one data point every 10 minutes
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


def holdout_data(merged_data):
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
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, svd_solver='full')
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index

    return(X_train_PCA, X_test_PCA)



def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold



#Daniele: will need to actually get the data from somewhere else at a later point in time...

###Define Mahalanobic Distance to classify data.
def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def train_PCA_model(X_train_PCA, X_test_PCA):
    #Daniele: train on the normal operating conditions
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)
    #compute covariance matrix
    covariance_matrix, inv_cov_matrix = cov_matrix(data_train)
    mean_distr = data_train.mean(axis=0)

    #calculate the Mahalanobis distance for the training data defining “normal conditions”,
    #and find the threshold value to flag datapoints as an anomaly.
    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    threshold = MD_threshold(dist_train, extreme=True)


    anomaly = pd.DataFrame()
    anomaly['Mob dist'] = dist_test
    anomaly['Thresh'] = threshold
    #Assign flags, marking data as anomalous or not anomalous
    # If Mob dist above threshold: Flag as anomaly.
    anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
    anomaly["Timestamps"] = X_test_PCA.index

    return anomaly






