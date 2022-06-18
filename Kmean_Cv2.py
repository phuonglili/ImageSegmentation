from cmath import pi
from tkinter import image_names
import numpy as np
import cv2
image = cv2.imread('input/img.jpg')
cv2.imwrite('output/goc.jpg',image) 
for i in range(1,10):
    image = cv2.imread('input/img.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.85)
    k = 10
    retval, labels, centers = cv2.kmeans(pixel_vals, i, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
    cv2.imwrite('output/'+str(i)+'.jpg', segmented_image)
    print("K = ",i)
    print(centers)
    print("---------------------------------------")
