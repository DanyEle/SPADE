import pandas as pd
from sklearn import preprocessing
import os
import numpy as np


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