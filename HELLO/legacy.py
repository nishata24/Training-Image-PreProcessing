import os
import numpy as np
import cv2 as cv

img_path = './img/'
save_folder = './img_crop/'

def crop_image(img):
    # crop the image
    print(img.shape)
    a = np.random.randint(0, 1540)
    b = np.random.randint(0, 789)
    cropped_img = img[0:a, 0:b]
    return cropped_img

def crop_image2(img):
    # randomly crop the image by rescaling it to a new size
    
    # get the image size
    h, w = img.shape[:2]
    #randomly crop the image by applying a multiple constant (<1) to the height and width
    uh= np.random.uniform(0, .15)
    uhc = 1 - uh

    uw = np.random.uniform(.1, .3)
    uwc = 1 - uw


    lh = int(h * uh)
    hh = int(h * uhc)
    
    lw = int(w * uw)
    hw = int(w * uwc)

    
    # crop the image to new h and w
    cropped_img = img[lh:hh, lw:hw]
    # resize the image to the original size

    # resized_img = cv.resize(cropped_img, (w, h))

    return cropped_img


# loop through all the images in the folder
for filename in os.listdir(img_path):
    # read the image
    img = cv.imread(img_path + filename, cv.IMREAD_UNCHANGED)

    # crop the image
    cropped_img = crop_image2(img)

    # save the cropped image
    cv.imwrite(save_folder + filename, cropped_img)


# crop_image(img, 1540, 789)