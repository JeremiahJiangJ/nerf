import cv2
import numpy as np
color_dict = {'red': np.uint8([[[0, 0, 255]]]),
              'green' : np.uint8([[[0 ,255, 0]]]),
              'blue' : np.uint8([[[255, 0, 0]]]),
              'white' : np.uint8([[[255, 255, 255]]]),
              'black' : np.uint8([[[0, 0, 0]]])}

hsv_dict = {key: cv2.cvtColor(color_dict[key], cv2.COLOR_BGR2HSV) for key in color_dict}

for key in hsv_dict:
    print(f"{key}:{hsv_dict[key]}")

hsv_color_dict = {key : [[hsv_dict[key][0][0][0] - 10, 100, 100], [hsv_dict[key][0][0][0] + 10, 255, 255]] for key in hsv_dict}

for key in hsv_color_dict:
    print(f"{key}:{hsv_color_dict[key]}")

see = np.uint8([[[21, 19, 0]]])
seehsv = cv2.cvtColor(see, cv2.COLOR_BGR2HSV)
print(seehsv)