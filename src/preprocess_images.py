# convert image to grayscale image
import cv2
import numpy as np
import imutils
from functools import reduce
DEBUG = 0

for i in range(90):
	img = cv2.imread(f"../../ISIC-Archive-Downloader/Data/Images/ISIC_{i:07}.jpeg")
	img = cv2.resize(img, (512,512))

	# threshholding in hsv
	# combine 2 masks cause red in hsv is both in 0-10 and 170-180
	converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower = np.array([0,40,20], dtype='uint8')
	upper = np.array([30,255,170], dtype='uint8')
	skinMask1 = cv2.inRange(converted, lower, upper)
	lower = np.array([170,40,20], dtype='uint8')
	upper = np.array([180,255,170], dtype='uint8')
	skinMask2 = cv2.inRange(converted, lower, upper)
	skinMask = cv2.add(skinMask1, skinMask2)
	if DEBUG:
		cv2.imshow("mask", skinMask)
		cv2.waitKey(250)

	# do some rough filtering
	kernel = np.ones((5,5),np.uint8)
	if DEBUG:
		cv2.imshow("mask", opening)
		cv2.waitKey(250)
	median = cv2.medianBlur(skinMask,5)
	if DEBUG:
		cv2.imshow("mask", median)
		cv2.waitKey(250)
	opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
	if DEBUG:	
		cv2.imshow("mask", opening)
		cv2.waitKey(250)

	# finding contours of objects
	cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# combine contours with biggest area 
	# (maybe can finetune 1000 based on number of contours, 
	# more contours -> smaller threshold. happens when skin marks are irregular)
	combinedCnt = np.array([], dtype=np.uint8).reshape(0,1,2)
	cnts_area = [cv2.contourArea(cnt) for cnt in cnts]
	cnts_big = [cnt for cnt in cnts if cv2.contourArea(cnt) > 900]
	for cnt in cnts_big:
		combinedCnt = np.concatenate((combinedCnt,cnt))

	# create bounding box from combined contours
	if(combinedCnt.shape[0] == 0):
		crop_img = img
	else:
		x,y,w,h = cv2.boundingRect(combinedCnt)
		crop_img = img[y:y+h,x:x+w]

	# crop and scale
	crop_img = img[y:y+h,x:x+w]
	scale_img = cv2.resize(crop_img, (512,512))

	if DEBUG:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow("mask", img)
		cv2.waitKey(250)

	if not DEBUG:
		cv2.imshow("EndImage", scale_img)
		cv2.waitKey(100)

cv2.waitKey(0)	