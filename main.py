from glob import glob
from image_utils import utils



def main():
    util=image_utils()
    test_imgs=sorted(glob("test/*"))
    util.improve_images(test_imgs)
    high_light_imgs=sorted(glob("high/*"))
    predicted_imgs=sorted(glob("predicted/*"))
    MSE,PSNR=util.AVG_MSE_PSNR(high_light_imgs,predicted_imgs)  
    print("MSE:",MSE,"PSNR",PSNR)                                         #  MSE: 101.81513564814817 PSNR 28.07976056684471
    

if __name__=="__main__":
    main()
