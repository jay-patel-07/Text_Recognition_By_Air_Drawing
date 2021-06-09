# import all necessary libraries
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

# load the model 
model = load_model('model_hand.h5')

# capture the live video
cap = cv2.VideoCapture(0);

# store the captured video in variable
_, old_frame = cap.read()
# a frame full of black-color
mask1 = np.zeros_like(old_frame)

# count the number of points on frame
counter = 0

# variable to store all the written letters
st=""
anss=""
while (True):
	
	# for every fraction of second, read the video
	ret, frame = cap.read()
	
	# all for printing 'AI Project' on screen
	cv2.rectangle(frame, (0, 0), (640, 55), (50, 50, 50), -1)
	
	font = cv2.FONT_HERSHEY_SIMPLEX

	# hsv in order to detect the object, one can use different value too
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
	
	# upper_bound and lower_bound of the hsv, just to identify the object clearly
	lb = np.array([0, 216, 136])
	ub = np.array([255, 255, 255])
	mask = cv2.inRange(hsv, lb, ub)
	
	# bitwise_and operation of the frame and frame, where mask is given
	res = cv2.bitwise_and(frame, frame, mask=mask)
	
	# find the countours
	edged = cv2.Canny(res, 30, 200)
	contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	img = cv2.drawContours(frame, contours, 0, (0, 255, 0), 0)
	flag=0
	# check for all the contours, to find the one we need
	for i in contours:
	
		# flag is set to 1, just in case when we detect contour
		flag=1
		
		# make the countour of rectangular shape
		(x, y, w, h)=cv2.boundingRect(i)
		# if the area of rectangle is less than 39 sq px, we don't require that contour
		if(cv2.contourArea(i)<39):
			continue;
		cnt = i
		
		# in order to find the center of rectangle
		M = cv2.moments(cnt)
		if M['m00'] != 0:
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])
			
			# draw the point where center is detected, and each time join previous point with current center
			if counter == 0:
				cx1 = cx
				cy1 = cy
			if counter == 0:
				counter += 1
				img = cv2.drawContours(frame, contours, 0, (0, 255, 0), 0)
			if(cx>=0 and cx<=100 and cy>=0 and cy<=100):
				# case when one wants to clear everything written on screen
				mask1 = np.zeros_like(old_frame)
				break
			
			# mask1 to store the shape written by joining all points
			mask1 = cv2.line(mask1, (cx, cy), (cx1, cy1), (0.0, 255), 2)
			cx1 = cx
			cy1 = cy
		
		# do the addition operation of frame and mask1
		img = cv2.add(frame, mask1)
		
	img = cv2.add(frame, mask1)
	
	# flip is needed because everything we write is the mirror image
	img=cv2.flip(img, 1)
	cv2.putText(img, 'AI Project', (200, 40), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
	cv2.putText(img, 'Clear All', (550, 85), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.rectangle(img, (480, 430), (640, 460), (50, 50, 50), -1)
	cv2.putText(img, st, (500, 455), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
	if(flag==0):
		# when there is no contour detected, then make counter as null
		counter=0
	
	mask2=cv2.flip(mask1, 1)
	
	if(cv2.waitKey(1) & 0xFF==ord('f')):
		st=""
		gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
		edged = cv2.Canny(gray, 30, 200)

		contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		ct=0
		crop=[]
		seq=[]
		j=0
		mx=0
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)
			if(w*h>19000):
				crop.append(mask2[y-20:y+h+20, x-20:x+w+20])
				seq.append([x, j])
				j+=1
		seq.sort()
		anss=""
		for ii in range(j):
			# convert the color img to gray scale of mask2
			print(seq)
			g=cv2.cvtColor(crop[seq[ii][1]], cv2.COLOR_BGR2GRAY)
			# resize the width and height to 28*28, because the model is trained with this size
			rr=cv2.resize(g, (28, 28), interpolation=cv2.INTER_AREA)
			
			# threshold the resized image
			_2, img_thresh2 = cv2.threshold(rr, 0, 255, cv2.THRESH_BINARY)
			new=np.array(img_thresh2).reshape(-1, 28, 28, 1)
			
			# now that's it, just predict
			p2=model.predict(new)
			
			chara2=np.argmax(p2)+ord('A')
			#print(chr(chara2))
			anss=anss+chr(chara2)
		print(anss)
		st=st+anss
	
		# for clearing the screen
		mask1 = np.zeros_like(old_frame)
	cv2.imshow("AI Project", img)
		
	# in order to quit the program, press 'q'
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

# release all the opened frames/ windows followed by destroying 'em
cap.release()
cv2.destroyAllWindows()

# store the string=st to a file named output.txt
sys.stdout = open("output.txt", "w")
print(st)
sys.stdout.close()
