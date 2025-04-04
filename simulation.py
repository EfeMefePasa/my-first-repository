#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
#Model / data parameters
num_classes = 10
input_shape = (28,28,1)

# Load the data and split it between train and test sets
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

#Scale images to the [0,1] range
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

#make sure images have shape (28,28,1)
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
#print("x_train shape:", x_train.shape)
#print(x_train.shape[0], "train samples")
#print(x_test.shape[0], "test samples")

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Build the model with keras sequential API

model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes,activation="softmax"),
         ]
    )
#model.summary()

# Padding is on valid meaning no padding and the stride parameter is on (1,1) meaning that we go for one pixel in each direction.
# Maxpool2d has a pool_size of 2 and a stride as large as the pool size meaning 2 

# Train the model
batch_size = 128
epochs=10
#myadam=keras.optimizers.Adam(learning_rate=0.1)
#myadam=keras.optimizers.Adam(learning_rate=0.00001)
#Default adam has a learning rate r=0.001
model.compile(loss="categorical_crossentropy", optimizer="adam" , metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model
score= model.evaluate(x_test,y_test,verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#model.save("Documents\DeepLearning.h5")

#Model simulieren und jetzt Filter-größe und Hyperparameter learning rate and dropout verändern damit
# sind die lernrate und der Dropout gemeint.




       