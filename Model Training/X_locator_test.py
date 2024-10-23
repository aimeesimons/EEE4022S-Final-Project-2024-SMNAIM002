import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import levenberg_marquardt as lm
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error

original_stout = sys.stdout
#lines = ['Central_Southern_400kV','NE_Central_400kV','NE_Central_400kV_c','NE_Central_400kV S1','']
directory = "\Fault Localisation\\Tester Model\\Line Lengths.csv"
df = pd.read_csv(directory)
lines = df['Line'].values
orderOfTypes = orderOfTypes =['ABC','ABG','AB','ACG','AC','AG','BCG','BC','BG','CG']
for line in lines:
    with open(f"Tests\\Test_Location_{line}_test2.txt", 'w',encoding='utf-8') as f:  #save output to textfile
        sys.stdout = f
        print(line)
        model = tf.keras.models.load_model(f"LineModels_new\\{line}\\{line}_10.keras") #load model 
        with open(f"Location Data\\{line}\\y_locate_{line}_testing.pickle", 'rb') as f:
            y_test = pickle.load(f)
        for fault in orderOfTypes: #test for each type of faults
            with open(f"LocationData\\{line}\\X_locate_input_test_{fault}.pickle", 'rb') as f: #get testing datasets
                X_test = pickle.load(f)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            print(fault)
            # Predict fault location
            predicted_fault_location = model.predict(X_test) #make predictions
            for i, distance in enumerate(predicted_fault_location):
                print(f"Predicted: ", distance[0], "Actual: ", y_test[i], "Difference (km)", (np.abs(distance[0]-y_test[i]))) #display predictions and km difference
            mse_values = (predicted_fault_location - y_test) ** 2
            # Find the maximum MSE
            mse = mean_squared_error(y_test,predicted_fault_location)
            print("Mean Squared Error: ", mse)


