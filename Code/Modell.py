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
X_train = np.load('drive/MyDrive/split_data/TrainX.npy')
y_train = np.load('drive/MyDrive/split_data/TrainY.npy')
X_test = np.load('drive/MyDrive/split_data/TestX.npy')
y_test = np.load('drive/MyDrive/split_data/TestY.npy')
X_val = np.load('drive/MyDrive/split_data/ValX.npy')
y_val = np.load('drive/MyDrive/split_data/ValY.npy')


#keras model
model_seq = keras.models.Sequential()
model_seq.add(keras.layers.Dense(2000, activation='sigmoid', input_shape= [9296,]))
model_seq.add(keras.layers.Dense(100, activation='sigmoid'))
model_seq.add(keras.layers.Dense(10, activation="softmax"))

#info model
print("Model Summery: ", model_seq.summary())
print("Model Layers: ", model_seq.layers)


hidden1 = model_seq.layers[1]
weights, biases = hidden1.get_weights()
print("Weights: ", weights)
print("Biases Shape: ", biases.shape)

#model compiling
opt = tf.keras.optimizers.SGD(learning_rate=0.03)
model_seq.compile(optimizer=opt, loss= "sparse_categorical_crossentropy", metrics=['accuracy'])

#Training and evaluating model
history = model_seq.fit(X_train,y_train, epochs= 10, validation_data=(X_test,y_test), batch_size=16)
print("fit history: ", history)


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.xlabel("Epochen")
plt.ylabel("Ergebnis")
plt.gca().set_ylim(0,1)
plt.show

model_seq.evaluate(X_val, y_val)

model_seq.save('drive/MyDrive/modelSeqSGDperfect')
