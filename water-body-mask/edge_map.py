import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_name = "kentridgeparkpond3"
image_path = f"./images/{image_name}.jpg"
hsv_savepath = "./hsv/"
grayscale_savepath = "./grayscale/"

if os.path.isdir(hsv_savepath) == False:
    os.mkdir(hsv_savepath)
if os.path.isdir(grayscale_savepath) == False:
    os.mkdir(grayscale_savepath)

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_grayscale, (3,3), 0)

cv2.imwrite(os.path.join(hsv_savepath, f"{image_name}_hsv.jpg"), img_hsv)
cv2.imwrite(os.path.join(grayscale_savepath, f"{image_name}_gs.jpg"), img_grayscale)
cv2.imwrite(os.path.join(grayscale_savepath, f"{image_name}_gs_blur.jpg"), img_blur)

edge = cv2.Canny(img_blur, 90, 100)

fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=150)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(edge, cmap='gray')
plt.show()

#contours = cv2.findContours(img_grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
#ccs = np.zeros(img_grayscale.shape, dtype = np.uint8)
#for i, contour in enumerate(contours):
#    cv2.drawContours(ccs, contours, i, color = 255, thickness=-1)
#cv2.imwrite("./contours.jpg", ccs)