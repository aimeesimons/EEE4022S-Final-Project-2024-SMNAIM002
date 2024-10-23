import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


lengths = pd.read_csv("\\Fault Localisation\\Tester Model\\Line Lengths.csv")
lines = lengths['Line'].values #get line names
for line in lines:
    print(line)
    matching_row = lengths[lengths['Line'] == line]
    corresponding_value = matching_row[' Length(Km)'].values[0]
    length = float(corresponding_value)
    with open(f"LocationData3\\{line}\\X_locate_input_noNoise.pickle", 'rb') as f:
        X1 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input10.pickle", 'rb') as f:
        X2 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input20.pickle", 'rb') as f:
        X3 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input30.pickle", 'rb') as f:
        X4 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input40.pickle", 'rb') as f:
        X5 = pickle.load(f)
    
    with open(f"LocationData3\\{line}\\X_locate_input50.pickle", 'rb') as f:
        X6 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input60.pickle", 'rb') as f:
        X7 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input70.pickle", 'rb') as f:
        X8 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input80.pickle", 'rb') as f:
        X9 = pickle.load(f)

    with open(f"LocationData3\\{line}\\X_locate_input90.pickle", 'rb') as f:
        X10 = pickle.load(f)


    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    X5 = np.array(X5)
    X6 = np.array(X6)
    X7 = np.array(X7)
    X8 = np.array(X8)
    X9 = np.array(X9)
    X10 = np.array(X10)

    X = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10), axis=0) #create dataset with 10100 datasamples

    with open(f"Location Data\\{line}\\y_locate_{line}_training.pickle", 'rb') as f:
        y = pickle.load(f)
    y = y*10
    X, y = shuffle(X,y)
    X = np.array(X)
    y = np.array(y)

    X_train = X
    y_train = y

    with open(f"LocationData3\\{line}\\X_locate_input_test_ABC.pickle", 'rb') as f: #initial testing 
        X_test = pickle.load(f)
    with open(f"Location Data2\\{line}\\y_locate_{line}_testing.pickle", 'rb') as f:
        y_test = pickle.load(f)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Define the model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 6)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer for regression
    ])

    # Compile the model
    callback = callbacks.EarlyStopping(monitor='loss', patience=20)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

   
    print("Train using Adam")
    history1= model.fit(X_train,y_train, epochs=700, batch_size=32)

    model.save(f"LineModels_new\\{line}\\{line}_10.keras")

    plt.plot(history1.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend('Train', loc='upper right')
    plt.show()


    # Predict fault location
    predicted_fault_location = model.predict(X_test)
    for i, distance in enumerate(predicted_fault_location):
        print("Predicted: ", distance[0], "Actual: ", y_test[i], "Error (%)", (np.abs(distance[0]-y_test[i])/length)*100)
    mse = mean_squared_error(y_test, predicted_fault_location)
    print("MSE: ", mse)


