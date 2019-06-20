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
from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix

ROOT_PATH = "../../"


train_directory_path = ROOT_PATH + 'train_new'

test_directory_path = ROOT_PATH + 'test_new'


width = 224
height = 224
batch_size = 16


test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(
    test_directory_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

pred_generator = test_datagen.flow_from_directory(
    test_directory_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

step_size_validate = validation_generator.n // validation_generator.batch_size
# Define Top2 and Top3 Accuracy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)



model = load_model('new_mdl.h5', custom_objects={'top_3_accuracy': top_3_accuracy, 'top_2_accuracy': top_2_accuracy})

labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
# evaluated = model.evaluate_generator(validation_generator, step_size_validate, verbose=1)

# pred = model.predict_generator(pred_generator, step_size_validate, verbose=1)
# y_pred = np.argmax(pred, axis=1)
# print(y_pred)
# predicted_class_indices=np.argmax(pred,axis=1)
# predictions = [labels[k] for k in predicted_class_indices]

# print(predictions)
correct = 0
incorrect = 0
correct_benign_malignant = 0
incorrect_benign_malignant = 0

malignant = ['MEL', 'BCC', 'BKL']
benign = ['NV', 'AKIEC', 'DF', 'VASC']
label_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

print(labels)

# print('evaluation:')
# print(model.metrics_names)
# print(evaluated)

y_pred = []
list_of_labels = os.listdir(test_directory_path)
for label in list_of_labels:
    current_label_dir_path = os.path.join(test_directory_path, label)
    list_of_images = os.listdir(current_label_dir_path)
    for image in list_of_images:
        current_image_path = os.path.join(current_label_dir_path, image)
        img = load_img(current_image_path, target_size=(224, 224))
        x = img_to_array(img) / 255.
        prediction = model.predict(x.reshape((1, 224, 224, 3)))
        prediction = prediction[0]
        temp = list(prediction)
        index_max_value = temp.index(max(temp))
        y_pred = y_pred + [index_max_value]
        
        # print(label, labels[index_max_value])
        if label == labels[index_max_value]:
            correct += 1
        else:
            incorrect += 1

        if labels[index_max_value] in malignant:
            if label in malignant:
                correct_benign_malignant += 1
            else:
                incorrect_benign_malignant += 1
        else:
            if label in benign:
                correct_benign_malignant += 1
            else:
                incorrect_benign_malignant += 1

print("accuracy in 7 classes:")
print(correct / (incorrect + correct))
print("accuracy in 2 classes:")
print(correct_benign_malignant / (incorrect_benign_malignant + correct_benign_malignant))

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=label_names))

