# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:49:38 2018

@author: S795641
"""

import keras
import os
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

#training paths defined
training_path = "C:/Users/Renz/Documents/Train_PreTrain/ClassifyingPretrained/frames/"
smile_path = training_path + "smile/"
serious_path = training_path + "serious/"


train_img = []
train_y = []

#loading images for drawings
for i in os.listdir(smile_path):
    temp_img = image.load_img(smile_path + i, target_size=(224,224))
    temp_img = image.img_to_array(temp_img)
    train_img.append(temp_img)
    train_y.append('smile')

#loading images for engraving
for i in os.listdir(serious_path):
    temp_img = image.load_img(serious_path + i, target_size=(224,224))
    temp_img = image.img_to_array(temp_img)
    train_img.append(temp_img)
    train_y.append('serious')


train_img = np.array(train_img)
train_img = preprocess_input(train_img)


#defining VGG16 model to extract features.
def vgg16_model(img_rows, img_cols, channel=-1, num_classes=None):
    model = VGG16(weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    x = Dense(num_classes, activation='softmax')(model.output)
    model = Model(model.input, x)
    for layer in model.layers[:8]:
        layer.trainable = False
    #change the learning rate of pretrained model to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
    return model


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
model = vgg16_model(img_rows, img_cols, channel, num_classes)
model.summary()



model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_Valid, Y_valid))

#saving the model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#saving the model weights
model.save_weights("model.h5")
print("Saved model to the disk")

#predictions = model.predict(test_img, batch_size=batch_size, verbose=1)
#
#from sklearn.metrics import log_loss
#score = log_loss(test_y, predictions)