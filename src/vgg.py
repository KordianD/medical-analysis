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
from keras.models import Model
from keras.optimizers import Adam

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

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

for layer in model.layers[:-8]:
    layer.trainable=False

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_validate = validation_generator.n // validation_generator.batch_size

checkpoint = keras.callbacks.ModelCheckpoint('checkpoint.check', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=2, 
                    validation_data=validation_generator,
                    validation_steps=step_size_validate,
                    callbacks=callbacks_list)

model.summary()
model.save('new_model.h5')

print(model.metrics_names)
s = model.evaluate_generator(validation_generator, step_size_validate, verbose=1)
print(s)