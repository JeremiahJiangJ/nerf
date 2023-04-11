import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

video_path = "D:/nerf/opticalflow/opticalflow_videos/building_farneback_1.mp4"

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items(), key = lambda x:x[1], reverse=True))
    
    #for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette

count = 0
while cap.isOpened() and count < 1:
    ret, image = cap.read()
    if not ret:
        break
    count += 1
    print(f'{count}/{frame_count}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Reshape image
    temp = image.reshape(-1,3)
    #unique, counts = np.unique(image, axis = 0, return_counts = True)
    #print(len(unique))
    #print(counts)

    clt = KMeans(3)
    clt.fit(temp)
    show_img_compar(image, palette_perc(clt))
    mask = image.copy()



    
