from flask import Flask, render_template
from scripts.ML_PCA import *

app = Flask(__name__)


#Data stored in this dictionary upon initializing the application
#data["timestamps"] = Format: Date + Time.The timestamp of every single data acquisition
#data["Anomaly"] = Format: Boolean. Reports whether the current data acquisition is anomalous
#data["Thresh"] = Float. Current value of the threshold identified
#data["Mob Dist"] = Float. Distance of the current data acquisition from the trained model.
data = {}


@app.route('/',  methods=['GET'])
def show_plot():
    timestamps = list(data.get("Timestamps"))
    #Remove the useless timestamp from the string
    timestamps_fixed = [str(s).replace('Timestamp(', '') for s in timestamps]
    return render_template("index.html", timestamps=timestamps_fixed, distances=list(data.get("Mob dist")),
                           anomalies=list(data.get("Anomaly")), thresholds = list(data.get("Thresh")))


def initialize_data():
    #Firstly, load the dataset
    merged_data = load_data_in_data_frame()

    print("Data successfully loaded")
    #Then, split the data into train and test
    #Training data: non-anomalous data (steady-state conditions)
    #Then, given a test sample, we compute the Mahalanobis
    #distance to the non-anomalous data
    # and classify the test points as an “anomaly”
    # if the distance is above a certain threshold.
    X_train_PCA, X_test_PCA = holdout_data(merged_data)

    #In our case, we will use a pre-acquired sample of steady-state data
    #to train the model and newly acquired test data to check
    #if the current state is anomalous.
    anomaly_all_data = train_PCA_model(X_train_PCA, X_test_PCA)

    return anomaly_all_data

    #print("Model successfully created")
    #plot_anomaly = anomaly_all_data.plot(logy=True, figsize = (10,6), ylim = [1e-1,1e3], color = ['green','red'])



#Commands ran upon starting application
if __name__ == '__main__':
    anomaly_all_data = initialize_data()
    data["Mob dist"] = anomaly_all_data["Mob dist"]
    data["Thresh"] = anomaly_all_data["Thresh"]
    data["Anomaly"] = anomaly_all_data["Anomaly"]
    data["Timestamps"] = anomaly_all_data["Timestamps"]

    app.run(debug=True)
