from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.layers import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam

# rows in csv: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
# malignant: MEL, BCC, BKL
# benign: NV, AKIEC, DF, VASC

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(7,activation='softmax')(x) #final layer with softmax activation

# Sizes of the scaled images
height = 450
width = 600

ROOT_PATH = "../../"

train_directory_path = ROOT_PATH + 'train_new'
test_directory_path = ROOT_PATH + 'test_new'

number_of_epochs = 2
batch_size = 16

train_data = ImageDataGenerator(
     rescale=1 / 255
)#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

#train_data = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_data.flow_from_directory(
    train_directory_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

validation_generator = test_datagen.flow_from_directory(
    test_directory_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

model=Model(inputs=base_model.input,outputs=preds)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_validate = validation_generator.n // validation_generator.batch_size


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint('.mdl.h5', save_best_only=True, monitor='val_loss', verbose=0, save_weights_only=False, mode='auto', period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
callbacks_list = [checkpoint, early_stopping, reduce_lr]
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10, 
                    validation_data=validation_generator,
                    validation_steps=step_size_validate,
                    callbacks=callbacks_list)


# print(model.metrics_names)

model = load_model(".mdl.h5")

s = model.evaluate_generator(validation_generator, step_size_validate, verbose=1)
print(s)
