import os
import csv
import shutil
from random import shuffle


ROOT_PATH = f"../../"
IMAGES_DIRECTORY = ROOT_PATH + "ISIC2018_Task3_Training_Input/"
LABELS_FILE = ROOT_PATH + "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"

TEST_DIRECTORY = "test_new"
TRAIN_DIRECTORY = "train_new"

IMAGE_MAPPING = dict()
IMAGE_MAPPING["benign"] = []
IMAGE_MAPPING["malignant"] = []

TRAIN_DATA_PERCENTAGE = 0.7

# malignant: MEL, BCC, BKL
# benign: NV, AKIEC, DF, VASC

def check_label(row):
    if row[1]=='1.0' or row[3]=='1.0' or row[5]=='1.0':
        IMAGE_MAPPING["malignant"].append(row[0] + ".jpg")
    else:
        IMAGE_MAPPING["benign"].append(row[0] + ".jpg")

def map_images():
    with open(LABELS_FILE) as file:
        reader = csv.reader(file)
        print(next(reader, None))  # skip the headers
        for row in reader:
            check_label(row)
    number_of_malignant_images = len(IMAGE_MAPPING["malignant"])
    print("How many malignant images " + str(number_of_malignant_images))

    number_of_benign_images = len(IMAGE_MAPPING["benign"])
    print("How many benign images " + str(number_of_benign_images))

map_images()

def shuffle_images():
    shuffle(IMAGE_MAPPING["benign"])
    shuffle(IMAGE_MAPPING["malignant"])

    train_data_benign_labels_amount = int(len(IMAGE_MAPPING["benign"]) * TRAIN_DATA_PERCENTAGE)
    train_data_malignant_labels_amount = int(len(IMAGE_MAPPING["malignant"]) * TRAIN_DATA_PERCENTAGE)

    print(len(IMAGE_MAPPING["malignant"]))

    train_data_benign_names = IMAGE_MAPPING["benign"][:train_data_benign_labels_amount]
    train_data_malignant_names = IMAGE_MAPPING["malignant"][:train_data_malignant_labels_amount]

    test_data_benign_names = IMAGE_MAPPING["benign"][train_data_benign_labels_amount:]
    test_data_malignant_names = IMAGE_MAPPING["malignant"][train_data_malignant_labels_amount:]

    copy_images(train_data_malignant_names, TRAIN_DIRECTORY, "malignant")
    copy_images(train_data_benign_names, TRAIN_DIRECTORY, "benign")
    copy_images(test_data_malignant_names, TEST_DIRECTORY, "malignant")
    copy_images(test_data_benign_names, TEST_DIRECTORY, "benign")

    copy_images(train_data_malignant_names, TRAIN_DIRECTORY, "malignant")
    copy_images(train_data_benign_names, TRAIN_DIRECTORY, "benign")
    copy_images(test_data_malignant_names, TEST_DIRECTORY, "malignant")
    copy_images(test_data_benign_names, TEST_DIRECTORY, "benign")

def copy_images(images_name, directory, label):
    for file in images_name:
            shutil.copy2(IMAGES_DIRECTORY + "/" + file, ROOT_PATH + "/" + directory + "/" + label)

shuffle_images()
print("OK")