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
X_train, X_test_full, y_train, y_test_full = train_test_split(x, Y, test_size=0.375, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_full, y_test_full, test_size=0.125)
print(X_train.shape)
print(X_test.shape)
print(X_val)
#keras
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val,y_val))

#keras model
model_seq = keras.models.Sequential()
model_seq.add(keras.layers.Dense(2000, activation='sigmoid', input_shape= [9296,]))
model_seq.add(keras.layers.Dense(10, activation='sigmoid'))
model_seq.add(keras.layers.Dense(10, activation="softmax"))

#info model
print("Model Summery: ", model_seq.summary())
print("Model Layers: ", model_seq.layers)


hidden1 = model_seq.layers[1]
weights, biases = hidden1.get_weights()
print("Weights: ", weights)
print("Biases Shape: ", biases.shape)

#model compiling
opt = tf.keras.optimizers.SGD(learning_rate=0.03 , momentum=0.9, clipnorm=1.0)
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model_seq.compile(optimizer=opt, loss= "sparse_categorical_crossentropy", metrics=['accuracy'])

#Training and evaluating model
history = model_seq.fit(X_train,y_train, epochs= 10, validation_data=(X_test,y_test), batch_size=16)
print("fit history: ", history)


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show

model_seq.evaluate(X_val, y_val)

#Prediction with Classification Report and Confusion Matrix
#prediction = model_seq.predict(X_val)
#prediction_max = np.argmax(prediction, axis = 1)
#print("Classification Report: ")
#print(classification_report(y_val, prediction_max))
#print("Confussion Matrix: ")
#print(confusion_matrix(y_val,prediction_max))

#Test image
pre321 = model_seq.call(x[321])
pre321_max = np.argmax(pre321, axis = 1)
print('Image as Test. Label: {} Prediction {}'.format(Y[321], pre321_max))
some_digit = x[321]
some_digit_image = some_digit.reshape(112,83)
plt.imshow(some_digit_image, cmap = mat.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

