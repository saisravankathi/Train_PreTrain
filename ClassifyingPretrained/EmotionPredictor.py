# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:55:03 2018

@author: S795641
"""
import os
import numpy as np
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

sgd = SGD()

classes_values = ["smile", "serious"]
classes_values.sort()
json_file = open("model.json","r")
loaded_model_json = json_file.read();
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model.h5")

model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

test_path = "C:/Keras/ClassifyingPretrained/frames/test/"

test_img = []
for i in os.listdir(test_path):
    temp_img = image.load_img(test_path + i, target_size=(224,224))
    temp_img = image.img_to_array(temp_img)
    test_img.append(temp_img)
    
test_img = np.array(test_img)
test_img = preprocess_input(test_img)


predictions = model.predict(test_img, batch_size=32, verbose=1)

#print("The classes are: ", classes_values)
count = 0
for k in predictions:
    index = list(k).index(max(k))
    print("The predction of "+os.listdir(test_path)[count]+" is "+ str(classes_values[index]))
    count +=1
