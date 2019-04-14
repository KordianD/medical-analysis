import json
import os

# with open('data.txt') as json_file:
#     data = json.load(json_file)

ROOT_PATH = f"../../ISIC-Archive-Downloader/Data/"
IMAGES_DIRECTORY = ROOT_PATH + "ProcessedImages/"
LABELS_DIRECTORY = ROOT_PATH + "Descriptions/"

TEST_DIRECTORY = "test"
TRAIN_DIRECTORY = "train"

BENIGN_DIRECTORY = "bening"
MALIGNANT_DIRECTORY = "malignant"

IMAGE_MAPPING = dict()
IMAGE_MAPPING["benign"] = []
IMAGE_MAPPING["malignant"] = []

for filename in os.listdir(IMAGES_DIRECTORY):
    print(filename)
    description_filename = filename.replace('.jpeg', '').replace('.png', '')

    with open(LABELS_DIRECTORY + description_filename) as json_file:
        description_json = json.load(json_file)

    if "benign_malignant" in description_json["meta"]["clinical"] and description_json["meta"]["clinical"][
        "benign_malignant"] and description_json["meta"]["clinical"]["benign_malignant"] != "indeterminate":

        if description_json["meta"]["clinical"]["benign_malignant"] == "indeterminate/malignant":
            IMAGE_MAPPING["malignant"].append(filename)
            continue

        if description_json["meta"]["clinical"]["benign_malignant"] == "indeterminate/benign":
            IMAGE_MAPPING["benign"].append(filename)
            continue

        print(description_json["meta"]["clinical"]["benign_malignant"])

        IMAGE_MAPPING[description_json["meta"]["clinical"]["benign_malignant"]].append(filename)

number_of_malignant_images = len(IMAGE_MAPPING["benign"])
print("How many malignant images " + str(number_of_malignant_images))

number_of_benign_images = len(IMAGE_MAPPING["malignant"])
print("How many benign images " + str(number_of_benign_images))

missing_images = len(os.listdir(IMAGES_DIRECTORY)) - number_of_benign_images - number_of_malignant_images
print("How many images are missing " + str(missing_images))
