# convert image to grayscale image
import cv2
import numpy as np
import imutils
import math
from functools import reduce
import os

# config

OUTPUT_SZ = (512, 512)
SAVE_PATH = f"../../ISIC-Archive-Downloader/Data/ProcessedImages/"
IMAGES_PATH = r"../../ISIC-Archive-Downloader/Data/Images"
DEBUG = 0
# video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'),20,OUTPUT_SZ,True)


for filename in os.listdir(IMAGES_PATH):
    print(IMAGES_PATH + "/" + filename)
    img = cv2.imread(IMAGES_PATH + "/" + filename)
    img_res = cv2.resize(img, OUTPUT_SZ)
    target_x = img.shape[1]
    target_y = img.shape[0]
    x_scale = target_x / OUTPUT_SZ[0]
    y_scale = target_y / OUTPUT_SZ[1]

    # threshholding in hsv
    # combine 2 masks cause red in hsv is both in 0-10 and 170-180
    converted = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 20], dtype='uint8')
    upper = np.array([30, 255, 170], dtype='uint8')
    skinMask1 = cv2.inRange(converted, lower, upper)
    lower = np.array([170, 40, 20], dtype='uint8')
    upper = np.array([180, 255, 170], dtype='uint8')
    skinMask2 = cv2.inRange(converted, lower, upper)
    skinMask = cv2.add(skinMask1, skinMask2)

    # do some rough filtering
    kernel = np.ones((5, 5), np.uint8)
    median = cv2.medianBlur(skinMask, 5)
    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)

    # finding contours of objects
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # combine contours with biggest area
    # (maybe can finetune 1000 based on number of contours,
    # more contours -> smaller threshold. happens when skin marks are irregular)
    combinedCnt = np.array([], dtype=np.uint8).reshape(0, 1, 2)
    cnts_area = [cv2.contourArea(cnt) for cnt in cnts]
    cnts_big = [cnt for cnt in cnts if cv2.contourArea(cnt) > 900]
    for cnt in cnts_big:
        combinedCnt = np.concatenate((combinedCnt, cnt))

    # create bounding box from combined contours
    if combinedCnt.shape[0] == 0:
        crop_img = img
    else:
        x, y, w, h = cv2.boundingRect(combinedCnt)
        (origLeft, origTop, origRight, origBottom) = (x, y, x + w, y + h)
        x = int(np.round(origLeft * x_scale))
        y = int(np.round(origTop * y_scale))
        xmax = int(np.round(origRight * x_scale))
        ymax = int(np.round(origBottom * y_scale))
        crop_img = img[y:ymax, x:xmax]
    scaled_img = cv2.resize(crop_img, OUTPUT_SZ)

    if DEBUG:
        cv2.imshow("Scaled", scaled_img)
        cv2.waitKey(150)
    else:
        cv2.imwrite(SAVE_PATH + filename, scaled_img)
# print(i)
# video.write(scaled_img)

# video.release()
cv2.waitKey(0)
