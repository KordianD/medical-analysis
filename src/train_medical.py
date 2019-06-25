from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.layers import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU

import pickle
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
# malignant: MEL, BCC, AKIEC
# benign: NV, DF, VASC, BKL

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

mobile = keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)

# Sizes of the scaled images
height = 450
width = 600
image_size = 224

ROOT_PATH = "../../"

train_directory_path = ROOT_PATH + 'TRAIN_FINAL/'
test_directory_path = ROOT_PATH + 'TEST_FINAL/'

number_of_epochs = 2
batch_size = 16

train_data = ImageDataGenerator()
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

print(len(model.layers))

# for layer in model.layers[:-20]:
#     layer.trainable = False

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
checkpoint = keras.callbacks.ModelCheckpoint('FINAL.h5', save_best_only=True, monitor='val_categorical_accuracy', verbose=1, save_weights_only=False, mode='max', period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                              patience=2, verbose=1, min_lr=0.00001, mode='max')
callbacks_list = [checkpoint, early_stopping, reduce_lr]

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=50, 
                    validation_data=validation_generator,
                    validation_steps=step_size_validate,
                    callbacks=callbacks_list)

# history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()