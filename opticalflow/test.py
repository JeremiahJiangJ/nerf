import cv2
import os
import numpy as np
video_path = "D:/nerf/opticalflow/opticalflow_videos/building_farneback_1.mp4"
boundary_path = "D:/nerf/opticalflow/boundaries/building_farneback_1_red1red2"
edge_path = "D:/nerf/opticalflow/edge/building_farneback"

color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}

color = 'red2'

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {frame_count}")
try:
    os.mkdir(boundary_path)
except OSError:
    pass

red1_lower = np.array(color_dict_HSV['red1'][1])
red1_upper = np.array(color_dict_HSV['red1'][0])

red2_lower = np.array(color_dict_HSV['red2'][1])
red2_upper = np.array(color_dict_HSV['red2'][0])

black_lower = np.array(color_dict_HSV['black'][1])
black_upper = np.array(color_dict_HSV['black'][0])

count = 0
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    count += 1
    print(f'{count}/{frame_count}')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(color_dict_HSV[color][1])
    upper = np.array(color_dict_HSV[color][0])
    mask_blue = cv2.inRange(hsv, np.array(color_dict_HSV['blue'][1]), np.array(color_dict_HSV['blue'][0]))
    mask_red1 = cv2.inRange(hsv, red1_lower, red1_upper)
    mask_red2 = cv2.inRange(hsv, red2_lower, red2_upper)
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    #mask = mask_red | mask_black 
    mask = mask_red1 | mask_red2 
    #mask = mask_red2
    #mask = ~mask_blue
    res = cv2.bitwise_and(image, image, mask = mask)

    kernel = np.ones((10,10), np.uint8)
    kernelopen = np.ones((20,20), np.uint8)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    #gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernelopen)
    #cv2.imshow('Gradient', gradient)
    cv2.imwrite(os.path.join(boundary_path, f'IMG_{str(count).zfill(4)}.png'), gradient)
    ### FIND LARGEST CONTOUR AND DISCARD THE REST ###
    #contours = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    #max_area = 0
    #max_contour_idx = 0
    #for i, contour in enumerate(contours):
    #    contour_area = cv2.moments(contour)['m00']
    #    if contour_area > max_area:
    #        max_area = contour_area
    #        max_contour_idx = i

    #largestCC_image = np.zeros(image.shape, dtype = np.uint8) ## blank image
    #cv2.drawContours(largestCC_image, contours, max_contour_idx, color = 255, thickness = -1) #draw in largset cc
    #cv2.imwrite(os.path.join(boundary_path, f'IMG_{str(count).zfill(4)}.png'), largestCC_image)

cv2.destroyAllWindows()
cap.release()
