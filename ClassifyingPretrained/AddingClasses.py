# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 19:51:46 2018

@author: Renz
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:49:38 2018

@author: S795641
"""

import os
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

#training paths defined
training_path = "C:/Users/Renz/Documents/Train_PreTrain/ClassifyingPretrained/frames/"
class_path = training_path + "rock/"
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)


train_img = []
train_y = []

#loading images for drawings
for i in os.listdir(class_path):
    temp_img = image.load_img(class_path + i, target_size=(224,224))
    temp_img = image.img_to_array(temp_img)
    train_img.append(temp_img)
    train_y.append('rock')

train_img = np.array(train_img)
train_img = preprocess_input(train_img)

le=LabelEncoder()
train_y=le.fit_transform(train_y)
train_y=to_categorical(train_y)
train_y = np.array(train_y)


from sklearn.model_selection import train_test_split

X_train, X_Valid, Y_train, Y_valid = train_test_split(train_img, train_y, test_size=0.2, random_state=42)

img_rows, img_cols = 224, 224
channel = 3,
num_classes = 2
batch_size = 16
nb_epoch = 10

#model
json_file = open("model.json","r")
loaded_model_json = json_file.read();
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
model = Sequential()
model.add(loaded_model)
weights_bak = model.layers[-1].get_weights()
print(np.array(weights_bak).shape)
nb_classes = model.layers[-1].output_shape[-1]
model.layers.pop()
model.add(Dense(nb_classes + 1, activation='softmax'))
weights_new = model.layers[-1].get_weights()

# copy the original weights back
weights_new[0][:, :-1] = weights_bak[0]
weights_new[1][:-1] = weights_bak[1]

# use the average weight to init the new class.
weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
weights_new[1][-1] = np.mean(weights_bak[1])

model.layers[-1].set_weights(weights_new)

model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

#model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_Valid, Y_valid))
#
##saving the model structure to JSON
#model_json = model.to_json()
#with open("model_new.json", "w") as json_file:
#    json_file.write(model_json)
##saving the model weights
#model.save_weights("model_new.h5")
#print("Saved model to the disk")

#predictions = model.predict(test_img, batch_size=batch_size, verbose=1)
#
#from sklearn.metrics import log_loss
#score = log_loss(test_y, predictions)