import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#%matplotlib inline
import numpy as np
from glob import glob
from PIL import Image,ImageOps
from model import ZeroDCE


class predictor:
    def __init__(self):
        models=self.from_path()
        self.models=models
    
    def plot_results(self,images,titles,figure_size=(12,12)):
        fig=plt.figure(figsize=figure_size)
        for i in range(len(images)):
            fig.add_subplot(1,len(images),i+1).set_title(titles[i])
            _=plt.imshow(images[i])
            plt.axis("off")
        plt.show()

    def infer(self,original_image):
        image=keras.utils.img_to_array(original_image)
        image=image.astype("float32")/255.0
        image=np.expand_dims(image,axis=0)
        output_image=self.models['zerodce'](image)
        output_image=tf.cast((output_image[0,:,:,:]*255),dtype=np.uint8)
        output_image=Image.fromarray(output_image.numpy())
        return output_image

    def from_path(self):
        models={}
        zerodce=ZeroDCE()
        print(zerodce.dce_model)
        zerodce.load_weights("weight.h5")
        models["zerodce"]=zerodce
        print(models)
        return models