# coding=utf-8

import os
import cv2
import numpy as np
from PIL import Image as im
from skimage.measure import label


def threshold_and_keep_largest_cc(datapath, savepath, lower=254, upper=255):
    if os.path.isdir(datapath) == False:
        print(f"{datapath} is INVALID")
        return

    if os.path.isdir(savepath) == False:
        os.mkdir(savepath)

    for file in os.listdir(datapath):
        
        ### THRESHOLD BINARIZE IMAGE ###
        image = cv2.imread(os.path.join(datapath, file), cv2.IMREAD_UNCHANGED)
        th, image_th = cv2.threshold(image, lower, upper, cv2.THRESH_BINARY)

        ### FIND LARGEST CONTOUR AND DISCARD THE REST ###
        contours = cv2.findContours(image_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        max_area = 0
        max_contour_idx = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.moments(contour)['m00']
            if contour_area > max_area:
                max_area = contour_area
                max_contour_idx = i

        largestCC_image = np.zeros(image_th.shape, dtype = np.uint8) ## blank image
        cv2.drawContours(largestCC_image, contours, max_contour_idx, color = 255, thickness = -1) #draw in largset cc
        cv2.imwrite(os.path.join(savepath, file), largestCC_image)




if __name__ == '__main__':
    #folder_name = 'map-mirroripad-cleaning'
    #datapath = f'./{folder_name}/{folder_name[4:]}/'
    lower_th, upper_th = 211, 255
    dataset_name = 'mirror'
    mask_gen = 'hetnet_msd'
    datapath = f'../datasets/{dataset_name}/refl_masks_{mask_gen}_og'
    savepath = f'../datasets/{dataset_name}/refl_masks_{mask_gen}_cleaned'
    threshold_and_keep_largest_cc(datapath, savepath, lower_th, upper_th)