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

train_directory_path = ROOT_PATH + 'TRAIN_FINAL/'
test_directory_path = ROOT_PATH + 'TEST_FINAL/'


width = 224
height = 224
batch_size = 20


test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(
    test_directory_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

gen = pred_generator = test_datagen.flow_from_directory(
    test_directory_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False,
    seed=42)

step_size_validate = validation_generator.n // validation_generator.batch_size
# Define Top2 and Top3 Accuracy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)



model = load_model('FINAL.h5', custom_objects={'top_3_accuracy': top_3_accuracy, 'top_2_accuracy': top_2_accuracy})

labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)

# eval = model.evaluate_generator(validation_generator, step_size_validate, verbose=1)
# print(eval)

#print(pred_generator.classes)
ground_truth = pred_generator.classes
true_labels = [labels[k] for k in ground_truth]
print(true_labels)

pred = model.predict_generator(pred_generator, step_size_validate, verbose=1)
y_pred = np.argmax(pred, axis=1)
#print(y_pred)

predicted_class_indices=np.argmax(pred,axis=1)
predictions = [labels[k] for k in predicted_class_indices]

print(predictions)

true_benign = 0
true_malignant = 0
false_benign = 0
false_malignant = 0

malignant = ['MEL', 'BCC', 'AKIEC']
benign = ['NV', 'BKL', 'VASC', 'DF']
label_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# print(labels)

# print('evaluation:')
# print(model.metrics_names)
# print(evaluated)

for predicted, true in zip(predictions, true_labels):
    if predicted in malignant:
        if true in malignant:
            true_malignant += 1
        else:
            false_malignant += 1
    else:
        if true in benign:
            true_benign += 1
        else:
            false_benign += 1

total = true_malignant + true_benign + false_benign + false_malignant
print("num of pictures:")
print(total)

# print("accuracy in 7 classes:")
# print(correct / (incorrect + correct))

print('Confusion Matrix for 7 classes')
print(confusion_matrix(ground_truth, predicted_class_indices))
print('Classification Report for 7 classes')
print(classification_report(ground_truth, predicted_class_indices, target_names=label_names))

print('Confusion matrix for 2 classes')
print('predicted   [ BEN   MAL ]')
print('actual BEN  [ {}   {} ]', true_benign, false_malignant)
print('actual MAL  [ {}   {} ]', false_benign, true_malignant)
print('')
print('Accuracy: {}', ((true_malignant+true_benign)/total) )
precision = true_malignant/(true_malignant+false_malignant)
recall = true_malignant/(true_malignant+false_benign)
print('Precision: {}', precision)
print('Recall: {}', recall)
print('F1: {}', 2*precision*recall/(precision+recall))


