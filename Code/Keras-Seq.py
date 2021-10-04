#import libraries
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.model_selection import cross_val_score
import matplotlib as mat
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')
file_path = 'drive/MyDrive/Daten/data.npy'
#load data
data = np.load(file_path)
np.random.shuffle(data)
print('All Data shape: ' ,data.shape)
Y = data[:,0]
x = data[:, 1:]
print('Label shape: ' ,Y.shape)
print('Image data shape: ' ,x.shape)

#Split the dataset
X_train, X_test_full, y_train, y_test_full = train_test_split(x, Y, test_size=0.35, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_full, y_test_full, test_size=0.1)
print(X_train.shape)
print(X_test.shape)
#keras
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = tf.data.Dataset.from_tensor_slices(X_val,y_val)

#keras model
model_seq = keras.models.Sequential()
model_seq.add(keras.layers.Flatten(input_shape= [1, 9296]))
model_seq.add(keras.layers.dense(300, activation="relu"))
model_seq.add(keras.layers.dense(100, activation="relu"))
model_seq.add(keras.layers.dense(10, activation="softmax"))

#info model
print("Model Summery: ", model_seq.summary())
print("Model Layers: ", model_seq.layers())


hidden1 = model_seq.layers[1]
weights, biases = hidden1.get_weights()
print("Weights: ", weights)
print("Biases Shape: ", biases.shape)

#model compiling
model_seq.compile(lose="sparse_categorical_crosstropy", optimizer="sgd", metrics=["accuracy"])

#Training and evaluating model
history = model_seq.fit(train_dataset, epochs= 30, validation_data= test_dataset)
print("fit history: ", history)


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gac().set_ylim(0,1)
plt.show

model_seq.evaluate(val_dataset)

#Prediction with Classification Report and Confusion Matrix
predition = model_seq.predict(X_val)
print("Classification Report: ")
print(classification_report(y_val, predition))
print("Confussion Matrix: ")
print(confusion_matrix(y_val,predition))

#Test image
pre321 = model_seq.predict(x[321])
print('Image as Test. Label: {} Predition {}'.format(Y[321], pre321 ))
some_digit = x[321]
some_digit_image = some_digit.reshape(112,83)
plt.imshow(some_digit_image, cmap = mat.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

