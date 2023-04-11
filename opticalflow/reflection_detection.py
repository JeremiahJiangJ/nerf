import os
import cv2
import numpy as np


image_path = "./lake.jpg"

img = cv2.imread(image_path)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = 100

ret, thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
print(hierarchy)
img_contours = np.zeros(img.shape)

cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
#save image
cv2.imwrite('./contours_125.png',img_contours) 
