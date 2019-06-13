import os
import csv
import shutil
from random import shuffle


ROOT_PATH = f"../../"
IMAGES_DIRECTORY = ROOT_PATH + "ISIC2018_Task3_Training_Input/"
LABELS_FILE = ROOT_PATH + "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"

TEST_DIRECTORY = "test_new"
TRAIN_DIRECTORY = "train_new"

# rows in csv: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
# malignant: MEL, BCC, BKL
# benign: NV, AKIEC, DF, VASC

skin_conditions = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
IMAGE_MAPPING = {k: [] for k in skin_conditions}

TRAIN_DATA_PERCENTAGE = 0.7

def check_label(row):
    idx = row[1:].index("1.0")
    IMAGE_MAPPING[skin_conditions[idx-1]].append(row[0]+".jpg")

def map_images():
    with open(LABELS_FILE) as file:
        reader = csv.reader(file)
        print(next(reader, None))  # skip the headers
        for row in reader:
            check_label(row)

def shuffle_images():
    for key, filenames in IMAGE_MAPPING.items():
        shuffle(filenames)
        train_data_labels_amount = int(len(filenames) * TRAIN_DATA_PERCENTAGE)
        train_data_names = filenames[:train_data_labels_amount]
        test_data_names = filenames[train_data_labels_amount:]

        copy_images(train_data_names, TRAIN_DIRECTORY, key)
        copy_images(test_data_names, TEST_DIRECTORY, key)


def copy_images(images_name, directory, label):
    if not os.path.exists(ROOT_PATH + "/" + directory + "/" + label):
        os.makedirs(ROOT_PATH + "/" + directory + "/" + label)
    for file in images_name:
            shutil.copy2(IMAGES_DIRECTORY + "/" + file, ROOT_PATH + "/" + directory + "/" + label)

map_images()
shuffle_images()
print("OK")