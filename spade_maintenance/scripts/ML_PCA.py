# Common imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#%matplotlib inline


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


 
    
def preprocess_data_pca(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, svd_solver='full')
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index

    return(X_train_PCA, X_test_PCA)



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


def plot_mahab_distance_square(dist_train):
    plt.figure()
    sns.distplot(np.square(dist_train),
                 bins = 10, 
                 kde= False);
    plt.xlim([0.0,15])
    
    
def plot_mahab_distance(dist_train):
    plt.figure()
    sns.distplot(dist_train,
                 bins = 10, 
                 kde= True, 
                color = 'green');
    plt.xlim([0.0,5])
    plt.xlabel('Mahalanobis dist')




