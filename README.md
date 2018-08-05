# Image Classification By Keras

This Code  it's about the image classification by neural networks by using the Keras Library  
 
 - Import All required library and packages
 - Download the dataset of Cat and Dog from kaggle 
 - Prepare The dataset in this step you must separate the images of dogs and cat by creating 2 folders one for dog image and 
     the other with cat images 
 - Prepare the test dataset and validation dataset
 - Create the ConvNet
 - Compile it 
 - Fit it to create the Model 
 - Make Predict of new input image 
 
 
 ## This is the code
 
 ```python
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation ,Dropout, Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras import optimizers
 
 ```python
train_data_dir='your training dataset path'
validation_data_dir='your validation database path'
 ```

 ```
 ```python
train_data_set=ImageDataGenerator().flow_from_directory(train_data_dir,target_size=(150,150),classes=['dog','cat'],batch_size=16)
validation_data_set=ImageDataGenerator().flow_from_directory(validation_data_dir,target_size=(150,150),classes=['dog','cat'],batch_size=16)

 ```
  ```python
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150,150,3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
 ```
 
  ```python
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

 ```
 
  ```python
model.fit_generator(train_data_set,steps_per_epoch=10,validation_data=validation_data_set,validation_steps=5,epochs=5)

 ```

