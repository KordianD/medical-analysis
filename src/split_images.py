import json
import os

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

with open('cancer_lookup.json') as json_file:
    cancer_lookup = json.load(json_file)


def extract_from_cancer_lookup_json(description_json):
    if "benign_malignant" in description_json["meta"]["clinical"] and "diagnosis" in description_json["meta"][
        "clinical"]:

        actual_diagnosis = description_json["meta"]["clinical"]["diagnosis"]

        if actual_diagnosis == "malignant":
            return "malignant"

        elif actual_diagnosis == "benign":
            return "benign"


def extract_from_bening_malignant_json_cell(description_json):
    if "benign_malignant" in description_json["meta"]["clinical"] and description_json["meta"]["clinical"][
        "benign_malignant"]:

        if description_json["meta"]["clinical"]["benign_malignant"] == "malignant":
            return "malignant"

        elif description_json["meta"]["clinical"]["benign_malignant"] == "benign":
            return "benign"


def add_filename_to_image_mapping(description_json, filename):
    extract_from_bening_malignant_json = extract_from_bening_malignant_json_cell(description_json)

    if extract_from_bening_malignant_json:
        IMAGE_MAPPING[extract_from_bening_malignant_json].append(filename)
        return

    extract_from_cancer_lookup = extract_from_cancer_lookup_json(description_json)

    if extract_from_cancer_lookup:
        IMAGE_MAPPING[extract_from_bening_malignant_json].append(filename)


for filename in os.listdir(IMAGES_DIRECTORY):
    description_filename = filename.replace('.jpeg', '').replace('.png', '')

    with open(LABELS_DIRECTORY + description_filename) as json_file:
        description_json = json.load(json_file)

    add_filename_to_image_mapping(description_json, filename)

number_of_malignant_images = len(IMAGE_MAPPING["malignant"])
print("How many malignant images " + str(number_of_malignant_images))

number_of_benign_images = len(IMAGE_MAPPING["benign"])
print("How many benign images " + str(number_of_benign_images))

missing_images = len(os.listdir(IMAGES_DIRECTORY)) - number_of_benign_images - number_of_malignant_images
print("How many images are missing " + str(missing_images))
