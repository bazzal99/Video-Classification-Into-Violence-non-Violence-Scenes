# -*- coding: utf-8 -*-
"""Video_Classification 94%

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rQjvNmTCf9IPEXLEvDu8bC9UaDpP14Ng
"""

from google.colab import drive
drive.mount('/content/drive/')

"""# New Section"""

pip install keras-metrics

import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras_metrics as km
 
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

data_dir = "/content/drive/My Drive/dataset1060"
img_height , img_width = 64, 64
seq_len = 60
classes = ["Non-Violence","Violence"]

def frames_extraction(path): 
    # Path to video file 
    myframe=seq_len
    vidObj = cv2.VideoCapture(path) 
    frames_list = []
    # Used as counter variable 
    count = 1
    # checks whether frames were extracted 
    success = 1
    fps = vidObj.get(cv2.CAP_PROP_FPS) 
    frame_count = vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = (frame_count) / fps
    second = 0
    vidObj.set(cv2.CAP_PROP_POS_MSEC, second * 1000) # optional
    success, image = vidObj.read() 
    milli=(duration*1000)/myframe
    while success and second<=duration: 
        second += milli/1000
        vidObj.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        image = cv2.resize(image, (img_height, img_width))
        frames_list.append(image)
       # cv2.imwrite("test/frame%d.jpg" % count,image)    
        count += 1
        success, image = vidObj.read()
    if len(frames_list)==(seq_len-1) :
        frames_list.append(frames_list[seq_len-2])
        print("-----------------------------------------")
    return frames_list

def create_data(input_dir):
    X = []
    Y = []
    i=0
    classes_list = os.listdir(input_dir)
    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        for f in files_list:
           if i%100==0 :
                print(i)               
           i=i+1
           frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
           if len(frames)==seq_len:
                X.append(frames)
                y = [0]*len(classes)
                y[classes.index(c)] = 1
                Y.append(y)
    return X, Y

X, Y = create_data(data_dir)
f=[]
p=[]
for i in range(len(X)) :
    data=X[i]
#   data.reverse()
    data = data[::-1]
    f.append(data)
    f.append(X[i])
    p.append(Y[i])
    p.append(Y[i])

f=np.asarray(f)
print(f.shape)
p=np.asarray(p)
print(p.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)
print(X.shape)
print(Y.shape)

model = Sequential()
model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width, 3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation = "softmax"))
 
model.summary()

opt = keras.optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
earlystop = EarlyStopping(patience=7)
callbacks = [earlystop]
history = model.fit(x = X_train, y = y_train, epochs=40, batch_size = 16, shuffle=True, validation_split=0.2, callbacks=callbacks)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
Y_test = np.argmax(y_test, axis = 1)
print(classification_report(Y_test, y_pred))