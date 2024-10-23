import tensorflow as tf
import pickle
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

classes_classification = ['AG','BG','CG','ABG','BCG','ACG','AB','BC','AC','ABC','NNNN']



X_classify_input = joblib.load("X_classify_input.pkl") #upload data

with open("y_classify.pickle", "rb") as f:
    y_classify = pickle.load(f)

y_classify = y_classify*2
print("Loaded Input")

# y_encode = [label_to_output.get(label,'Unknown') for label in y_classify]

label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(y_classify) #encode y so that it represents numbers from 0 to 10.


# Convert numerical labels to one-hot encoded labels
y_encoded = tf.keras.utils.to_categorical(numerical_labels) #convert to one hot encoding vectors.
print(label_encoder.classes_)
print("encoded y")

X_train, X_test, y_train, y_test = train_test_split(X_classify_input, y_encoded, test_size=0.3, random_state=42, shuffle=True) #split and shuffle data

print("Split")

model_2 = tf.keras.models.Sequential()

model_2.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = (127, 127, 6)))
model_2.add(tf.keras.layers.Activation("relu"))
model_2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


model_2.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model_2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model_2.add(tf.keras.layers.Flatten())
model_2.add(tf.keras.layers.Dense(64, activation ='relu'))

model_2.add(tf.keras.layers.Dense(11, activation='softmax'))


model_2.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'] )
history = model_2.fit(X_train,y_train, validation_split=0.1,batch_size=32, epochs=5) #train model


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

joblib.dump(model_2, "ClassificationModel_mixed_noise_2.joblib") #save model

y_pred = model_2.predict(X_test)
y_pred_one_hot = (y_pred>0.5).astype(int)
y_pred_indices = np.argmax(y_pred_one_hot, axis=1)
# Perform inverse transform to get the original string labels
y_pred_classes = label_encoder.inverse_transform(y_pred_indices)

#Test model
y_test = y_test.astype(int)
y_test_indices = np.argmax(y_test, axis=1)
y_test_classes = label_encoder.inverse_transform(y_test_indices)

cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes, average='macro')
recall = recall_score(y_test_classes, y_pred_classes, average='macro')
f1 = f1_score(y_test_classes, y_pred_classes, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')