import os
import cv2
import numpy as np
import fast_colorthief as fct
from PIL import Image
video_name = "building_farneback_1"
#video_name = "lake_farneback"

mask_name = "test_manual"
auto = False
video_path = f"D:/nerf/opticalflow/opticalflow_videos/{video_name}.mp4"
boundary_path = f"D:/nerf/opticalflow/test_reflection_generation/{video_name}_{mask_name}"
try:
    os.mkdir(boundary_path)
except OSError:
    pass
### GENERATING COLOR DICTIONARY ###
#color_dict_hsv = {'black': [[180, 255, 13], [0, 0, 0]],
#              'white': [[180, 18, 255], [0, 0, 231]],
#              'red1': [[180, 255, 255], [159, 20, 10]],
#              'red2': [[9, 255, 255], [0, 20, 10]],
#              'green': [[89, 255, 255], [36, 20, 10]],
#              'blue': [[128, 255, 255], [89, 20, 10]],
#              'yellow': [[36, 255, 255], [25, 20, 10]],
#              'purple': [[158, 255, 255], [129, 20, 10]],
#              'orange': [[24, 255, 255], [10, 20, 10]],
#              'gray': [[180, 18, 230], [0, 0, 40]]}

color_dict_hsv = {'black'  : [[0, 0, 0], [180, 255, 10]],
                  'white'  : [[0, 0, 255], [180, 18, 255]],
                  'red1'   : [[163, 20, 10], [180, 255, 255]],
                  'pink'   : [[135, 20 ,10], [163, 255, 255]],
                  'blue'   : [[82, 20, 10], [135, 255, 255]],
                  'green'  : [[41, 20, 10], [82, 255, 255]],
                  'yellow' : [[24, 20, 10], [41, 255, 255]],
                  'orange' : [[10, 20, 10], [25, 255, 255]],
                  'red2'   : [[0, 20, 10], [10, 20, 10]]}

color_dict_hsv = {'black'  : [[0, 0, 0], [180, 255, 10]],
                  'white'  : [[0, 0, 255], [180, 18, 255]],
                  'red1'   : [[163, 12, 10], [180, 255, 255]],
                  'pink'   : [[135, 12 ,10], [163, 255, 255]],
                  'blue'   : [[75, 12, 10], [135, 255, 255]],
                  'green'  : [[41, 12, 10], [75, 255, 255]],
                  'yellow' : [[24, 12, 10], [41, 255, 255]],
                  'orange' : [[10, 12, 10], [25, 255, 255]],
                  'red2'   : [[0, 12, 10], [10, 255, 255]]}

### END OF GENERATING COLOR DICTIOANRY ###

#identify most dominant color and create a mask to exclude it


#def create_mask(hsv, color):
#    if color == 'red':
#        mask1 = cv2.inRange(hsv, np.array(color_dict_hsv['red1'][1]), np.array(color_dict_hsv['red1'][0]))
#        mask2 = cv2.inRange(hsv, np.array(color_dict_hsv['red2'][1]), np.array(color_dict_hsv['red2'][0]))
#        mask = mask1 + mask2
#    else:
#        mask = cv2.inRange(hsv, np.array(color_dict_hsv[color][1]), np.array(color_dict_hsv[color][0]))
#    return mask

def create_mask(hsv, color):
    if color == 'red':
        mask1 = cv2.inRange(hsv, np.array(color_dict_hsv['red1'][0]), np.array(color_dict_hsv['red1'][1]))
        mask2 = cv2.inRange(hsv, np.array(color_dict_hsv['red2'][0]), np.array(color_dict_hsv['red2'][1]))
        mask = mask1 + mask2
    else:
        mask = cv2.inRange(hsv, np.array(color_dict_hsv[color][0]), np.array(color_dict_hsv[color][1]))
    return mask

def check_mask(color):
    img = cv2.imread("HSVWHEEL.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = create_mask(hsv, color)
    res = cv2.bitwise_and(img, img, mask=~mask)
    cv2.imshow("Everything but the mask", res)
    cv2.imshow("Image", img)
    cv2.waitKey(10000)
#check_mask('green')

def get_dominant_colors(img, color_count=2, quality=1, to_hsv=False):
    dominant_colors = fct.get_palette(img, color_count=color_count, quality=quality)
    res = []
    for color in dominant_colors:
        #print(color)
        lst_color = np.uint8([[list(color)]]) #convert result from tuple to 3d np array
        if to_hsv:
            hsv_color = cv2.cvtColor(lst_color, cv2.COLOR_RGB2HSV)
            res.append(hsv_color)
        else:
            res.append(lst_color)
    return res

def get_dominant_color_string(dominant_colors, hsv_color_dct):
    res = []
    for color in dominant_colors:
        #print(color)
        h,s,v = color[0][0][0], color[0][0][1], color[0][0][2]
        #print(f'HSV = {h,s,v}')
        for hsv_range in color_dict_hsv:
            lower = color_dict_hsv[hsv_range][0]
            upper = color_dict_hsv[hsv_range][1]
            lower_h, lower_s, lower_v  = lower[0], lower[1], lower[2]
            upper_h, upper_s, upper_v = upper[0], upper[1], upper[2]

            if (h >= lower_h and h <= upper_h) and (s >= lower_s and s <= upper_s) and (v >= lower_v and v <= upper_v):
                if(hsv_range != "black"):
                    res.append(hsv_range)
    return res

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {frame_count}")
count = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    count+=1
    h, w = img.shape[:2]

    dominant_colors = get_dominant_colors(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA), to_hsv=True)
    colors_to_mask = get_dominant_color_string(dominant_colors, color_dict_hsv) 

    cv2.imwrite(os.path.join(boundary_path, 'original', f'IMG_{str(count).zfill(4)}.png'), img)
    print(f'{count}/{frame_count}')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    refl_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if auto:
        
        for i in colors_to_mask:
            print(f'      Most dominant color :{i}')
            curr_mask = create_mask(hsv, i)
            refl_mask = refl_mask + create_mask(hsv, i)
    else:
        refl_mask = create_mask(hsv, 'red') + create_mask(hsv, 'pink') + create_mask(hsv, 'yellow')
    #print(img.shape)
    #print(refl_mask.shape)
    # KEEP THIS #
    #mask_dominant = create_mask(hsv, 'blue') + create_mask(hsv, 'black') + create_mask(hsv, 'green')
    #refl_mask2 = cv2.bitwise_not(mask_dominant)
    #print(f'Refl_mask2 shape :{refl_mask2.shape}')
    # KEEP THIS ^ #
    
    res = cv2.bitwise_and(img, img, mask=refl_mask)
    th, refl_mask = cv2.threshold(refl_mask, 254, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(boundary_path, 'reflection', f'IMG_{str(count).zfill(4)}_refl.png'),  refl_mask)
    cv2.imwrite(os.path.join(boundary_path, 'masked_out', f'IMG_{str(count).zfill(4)}.png'),  res)

 
    gradient_kernel = np.ones((15, 15), np.uint8)
    erosion_kernel = np.ones((11,11), np.uint8)
    #dilation_kernel = np.ones((5, 5), np.uint8)
    close_kernel = np.ones((15,15), np.uint8)
    gradient = cv2.morphologyEx(refl_mask, cv2.MORPH_GRADIENT, gradient_kernel)
    img_eroded = cv2.erode(gradient, erosion_kernel, iterations=1)
    #img_dilated = cv2.dilate(img_eroded, dilation_kernel, iterations=1)
    img_closed = cv2.morphologyEx(refl_mask, cv2.MORPH_CLOSE, close_kernel)
    cv2.imwrite(os.path.join(boundary_path, 'gradient', f'IMG_{str(count).zfill(4)}_grad.png'), gradient)
    cv2.imwrite(os.path.join(boundary_path, 'eroded', f'IMG_{str(count).zfill(4)}_eroded.png'), img_eroded)
    cv2.imwrite(os.path.join(boundary_path, 'closed', f'IMG_{str(count).zfill(4)}_closed.png'), img_closed)

    contours = cv2.findContours(img_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    max_area = 0
    max_contour_idx = 0
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)['m00']
        if contour_area > max_area:
            max_area = contour_area
            max_contour_idx = i

    largestcc_image = np.zeros(refl_mask.shape, dtype = np.uint8) ## blank image
    cv2.drawContours(largestcc_image, contours, max_contour_idx, color = 255, thickness = -1) #draw in largset cc
    #largestcc_image = cv2.morphologyEx(largestcc_image, cv2.MORPH_OPEN, np.ones((50,50),np.uint8))
    #largestcc_image = cv2.morphologyEx(largestcc_image, cv2.MORPH_CLOSE, np.ones((150,150), np.uint8))
    cv2.imwrite(os.path.join(boundary_path, 'largest', f'IMG_{str(count).zfill(4)}_largest.png'), largestcc_image)

    #Flood fill
    img_floodfill = largestcc_image.copy()
    floodfill_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, floodfill_mask, (0,0), 255)

    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    
    img_out = img_floodfill_inv + largestcc_image
    cv2.imwrite(os.path.join(boundary_path, 'refl_masks', f'IMG_{str(count).zfill(4)}.png'), img_out)


    
cap.release()




    