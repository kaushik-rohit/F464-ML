import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.morphology import medial_axis
from skimage import img_as_ubyte
from skimage.morphology import skeletonize, thin

dir_path = os.path.dirname(os.path.realpath(__file__))
train_path = dir_path + '/train_images/'

if __name__ == '__main__':
    
    train_images_path = dir_path+'/train_images'
    train_images = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]
    
    for file_name in train_images:
    
        img = cv2.imread(train_path + file_name, 0) #load image as grayscale
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)
        ret, ret_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #apply binary threshold
        cv2.fastNlMeansDenoising(img, img)
        img=img/255
        skel = thin(ret_img, max_iter=3)
        ret_img = img_as_ubyte(skel)
        resz_img = cv2.resize(ret_img, (64,64)) #resize it to 25*25 image
        
        #kernel1 = np.ones((3,3), np.uint8)
        #erosion = cv2.erode(resz_img, kernel1, iterations = 1)
        
        #kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        #closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2)
        ret, res_img = cv2.threshold(resz_img, 0, 255, cv2.THRESH_BINARY_INV)
        #res_img = cv2.filter2D(res_img,-1,kernel)
        
        cv2.imwrite(dir_path + '/pre/' + file_name, res_img)
        
