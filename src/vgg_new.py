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

mobile = keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
x = Dropout(0.15)(x)
predictions = Dense(7, activation='softmax')(x)

# Sizes of the scaled images
height = 450
width = 600
image_size = 224

ROOT_PATH = "../../"

train_directory_path = ROOT_PATH + 'train_new'
test_directory_path = ROOT_PATH + 'test_new'

number_of_epochs = 2
batch_size = 16

train_data = ImageDataGenerator()
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

#train_data = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_data.flow_from_directory(
    train_directory_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

validation_generator = test_datagen.flow_from_directory(
    test_directory_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

model=Model(inputs=mobile.input,outputs=predictions)

for layer in model.layers[:-23]:
    layer.trainable = False

# Define Top2 and Top3 Accuracy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01),loss='categorical_crossentropy',metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

step_size_train=train_generator.n//train_generator.batch_size
step_size_validate = validation_generator.n // validation_generator.batch_size




early_stopping = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint('new_mdl.h5', save_best_only=True, monitor='val_categorical_accuracy', verbose=1, save_weights_only=False, mode='max', period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                              patience=2, verbose=1, min_lr=0.00001, mode='max')
callbacks_list = [checkpoint, early_stopping, reduce_lr]

class_weight = {0: 7.,  #AKIEC
                1: 22., #BCC
                2: 60,  #BKL
                3: 45.,  #DF
                4: 1.,  #MEL
                5: 15.,  #NV
                6: 7.,} #VASC

# MEL :  6705
# NV :  514
# BCC :  327
# AKIEC :  1099
# BKL :  115
# DF :  142
# VASC :  1113

model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=100, 
                    validation_data=validation_generator,
                    validation_steps=step_size_validate,
                    callbacks=callbacks_list,
                    class_weight=class_weight)
