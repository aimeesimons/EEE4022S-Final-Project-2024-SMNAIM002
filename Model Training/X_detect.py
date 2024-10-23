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

classes_detect = ['Fault Detected', 'Fault Not Detected']



X_detect_input = joblib.load("X_detect_input_noisy_varied1.pkl") #upload data
with open("y_detect.pickle", "rb") as f:
    y_detect = pickle.load(f)
    
print("Loaded Input")


y_detect_edited = []
for y in y_detect:
    y_detect_edited.append([y])
values = array(y_detect_edited)
ordinal_encoder = OrdinalEncoder()
y_encode = ordinal_encoder.fit_transform(values)
print(ordinal_encoder.categories_) #encode y so that it either represents a 1 or 0.
print("encoded y")
X_detect_input, y_encode = shuffle(X_detect_input, y_encode)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X_detect_input, y_encode, test_size=0.3, random_state=42)

print("Split")

model_1 = tf.keras.models.Sequential()


model_1.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = (127, 127, 6)))
model_1.add(tf.keras.layers.Activation("relu"))
model_1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))




model_1.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model_1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


model_1.add(tf.keras.layers.Flatten())
model_1.add(tf.keras.layers.Dense(64, activation ='relu'))



model_1.add(tf.keras.layers.Dense(1)) 
model_1.add(tf.keras.layers.Activation("sigmoid"))

model_1.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ['accuracy'] )

history = model_1.fit(X_train,y_train,validation_split=0.1, batch_size=32, epochs=5)  #train model


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

joblib.dump(model_1, "DetectionModel_mixed_noise_2.joblib") #save model
  
#Test model
y_pred = model_1.predict(X_test)
y_pred_encoded = (y_pred > 0.5).astype(int)
y_pred_classes = ordinal_encoder.inverse_transform(y_pred_encoded.reshape(-1,1))
y_test_classes = ordinal_encoder.inverse_transform(y_test.reshape(-1,1))



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