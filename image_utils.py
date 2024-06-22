import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from math import log10,sqrt
from predictor import predictor 
from PIL import Image
import cv2

pd=predictor()
class utils:
    def __init__(self):
        return None
    
    def improve_images(self,f):
        for i in f:
            a=i.split("test\\")
            b=a[1].split(".png")
            img_name=b[0]
            img=Image.open(i)
            eimg=pd.infer(img)
            eimg_arr=keras.utils.img_to_array(eimg)
            eimg_arr=eimg_arr/255
            plt.imsave(f'predicted/{img_name}.png',eimg_arr)
        
        return True

    def AVG_MSE_PSNR(self,f,p): 
        avg_mse=0
        avg_psnr=0
        max_pixel = 255.0
        no_of_images=len(f)
        for i in range(no_of_images):
            img=cv2.imread(f[i])
            ei=cv2.imread(p[i])
            
            mse = np.mean((img - ei) ** 2) 
            if(mse == 0):
                psnr= 100
            else:
                psnr = 20 * log10(max_pixel / sqrt(mse)) 
            
            avg_mse+=mse
            avg_psnr+=psnr
        
        avg_mse/=no_of_images
        avg_psnr/=no_of_images
        return avg_mse,avg_psnr 
    
