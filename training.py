import tensorflow as tf
import numpy as np
import os


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

model = tf.keras.models.Sequential()
# 2D convolutional layer - 32 filters, kernel size 3 by 3, activation relu, 32 by 32 pixels, 3 channel rgb
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# flatten results
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
# 10 outputs due to 10 possible classes
# softmax is producing a result where if you add up all the individual neuron activations you will get 1 (100%)
# each neuron add how likely it is close to correct classification
model.add(tf.keras.layers.Dense(10, activation = "softmax"))
model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
# fit on training data with 10 epochs
model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test))
model.save("cifar_classifier.model")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




