import os
import cv2
import numpy as np
import fast_colorthief as fct

from PIL import Image
original_video_name ="DJI_0005"
optflow_video_name = f"{original_video_name}_farneback"
#video_name = "lake_farneback"
original_video_path = f"D:/nerf/opticalflow/videos/{original_video_name}.mp4"
optflow_video_path = f"D:/nerf/opticalflow/opticalflow_videos/{optflow_video_name}.mp4"

color_dict_hsv = {'black'  : [[0, 0, 0], [180, 255, 10]],
                  'white'  : [[0, 0, 255], [180, 18, 255]],
                  'red1'   : [[163, 12, 10], [180, 255, 255]],
                  'pink'   : [[135, 12 ,10], [163, 255, 255]],
                  'blue'   : [[75, 12, 10], [135, 255, 255]],
                  'green'  : [[41, 12, 10], [75, 255, 255]],
                  'yellow' : [[24, 12, 10], [41, 255, 255]],
                  'orange' : [[10, 12, 10], [25, 255, 255]],
                  'red2'   : [[0, 12, 10], [10, 255, 255]]}


def create_mask(hsv, color):
    if color == 'red':
        mask1 = cv2.inRange(hsv, np.array(color_dict_hsv['red1'][0]), np.array(color_dict_hsv['red1'][1]))
        mask2 = cv2.inRange(hsv, np.array(color_dict_hsv['red2'][0]), np.array(color_dict_hsv['red2'][1]))
        mask = mask1 + mask2
    else:
        mask = cv2.inRange(hsv, np.array(color_dict_hsv[color][0]), np.array(color_dict_hsv[color][1]))
    return mask

original_cap = cv2.VideoCapture(original_video_path)
optflow_cap = cv2.VideoCapture(optflow_video_path)
frame_count = int(optflow_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {frame_count}")
count = 0
keep_top_x_contours = 2

#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#prev_img = None
save = True
while True:
    ret_og, img_og = original_cap.read()
    ret_of, img_of = optflow_cap.read()

    if not ret_og or not ret_of or count > 500:
        break

    count += 1
    print(f'{count}/{frame_count}')

    h, w = img_og.shape[:2]

    img_of_gray = cv2.cvtColor(img_of, cv2.COLOR_BGR2GRAY)
    img_og_gray = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
    
    img_contours = np.zeros(img_og.shape[:2])

    
    #curr_img = img_og_gray
    #if prev_img is not None:
    #    disparity = stereo.compute(prev_img, curr_img)
    #    if save:
    #        cv2.imwrite(f"./debug_dji/disparity_maps/disparity_between_{count-1}_{count}.png", disparity)
    #prev_img = curr_img

    #Contour of optflow video
    hsv_of = cv2.cvtColor(img_of, cv2.COLOR_BGR2HSV)
    hsv_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2HSV)

    #refl_mask = np.zeros(img_of.shape[:2], dtype=np.uint8)
    black_mask = create_mask(hsv_of, 'black')
    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x))
    contours_cleaned = contours[-keep_top_x_contours:]
    cv2.drawContours(img_contours, contours_cleaned, -1, 255, -1)
    
    #Calculate color in contours_final, if there is a lot of white and blue, treat it as the sky
    final = np.zeros(img_og.shape[:2], np.uint8)
    refl_mask = np.zeros(img_og_gray.shape[:2], np.uint8)
    
    #consider using hsv

    for i, contour in enumerate(contours_cleaned):
        refl_mask[...] = 0
        cv2.drawContours(refl_mask, contours_cleaned, i, 255, -1)
        mean_hsv = cv2.mean(hsv_og, refl_mask)
        mean_bgr =  cv2.mean(img_og, refl_mask)
        print(f"Mean HSV= {mean_hsv}")
        print(f"Mean BGR = {mean_bgr}")
        b,g,r = mean_bgr[0], mean_bgr[1], mean_bgr[2]        
        h,s,v = mean_hsv[0], mean_hsv[1], mean_hsv[2] 
        if b < 200 and g < 200 and r < 200:
            cv2.imwrite(f'./debug_dji/refl_masks/refl_mask_{count}.png',refl_mask) 
        #if s > 50 and v < 240:
        #    cv2.imwrite(f'./debug_dji/refl_masks/refl_mask_{count}.png',refl_mask) 



    if save:
        cv2.imwrite(f'./debug_dji/originals/original_{count}.png',img_og)
        cv2.imwrite(f'./debug_dji/opticalflows/opticalflow_{count}.png',img_of)
        cv2.imwrite(f'./debug_dji/black_masks/black_mask_{count}.png', black_mask)
        cv2.imwrite(f'./debug_dji/contours/contours_{count}.png',img_contours) 
        

original_cap.release()
optflow_cap.release()

#while optflow_cap.isOpened() and count < 100:
#    ret, img = optflow_cap.read()
#    if not ret:
#        break
    
#    count+=1
#    h, w = img.shape[:2]
#    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#    img_contours = np.zeros(img.shape)

#    print(f'{count}/{frame_count}')
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#    refl_mask = np.zeros(img.shape[:2], dtype=np.uint8)
#    refl_mask = create_mask(hsv, 'black')
    
#    res = cv2.bitwise_and(img, img, mask=refl_mask)
#    #th, refl_mask = cv2.threshold(refl_mask, 254, 255, cv2.THRESH_BINARY)
#    contours, hierarchy = cv2.findContours(refl_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    contours = sorted(contours, key=lambda x:cv2.contourArea(x))
 
#    contours_final = contours[-keep_top_x_contours:]
#    cv2.drawContours(img_contours, contours_final, -1, (255,255,255), 1)
#    #save image
#    cv2.imwrite(f'./debug_dji/originals/original_{count}.png',img)
#    cv2.imwrite(f'./debug_dji/refl_masks/refl_mask_{count}.png', refl_mask)
#    cv2.imwrite(f'./debug_dji/contours/contours_{count}.png',img_contours) 
#    prev_img = img_gray
    
#cap.release()




    