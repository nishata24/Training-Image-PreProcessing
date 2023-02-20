import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import skimage
from skimage.color import rgba2rgb
from skimage import data


def crop_image(img):
    # crop the image
    print(img.shape)
    a = np.random.randint(0, 1540)
    b = np.random.randint(0, 789)
    cropped_img = img[0:a, 0:b]
    return cropped_img

#def flipping(img):
    #randomly take a horizontal flip
    if np.random.random() < 0.5:
        img = cv.flip(img, 1)

    #randomly take a vertical flip
    if np.random.random() < 0.5:
        img = cv.flip(img, 0)
    
    #randomly rotate the image by 90, 180, or 270 degrees
    if np.random.random() < 0.33:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    elif np.random.random() < 0.66 and np.random.random() > 0.33:
        img = cv.rotate(img, cv.ROTATE_180)
    else:
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    return img

def crop_image2(img):
    #img = flipping(img)
    
    # get the image size
    h, w = img.shape[:2]
    #randomly crop the image by applying a multiple constant (<1) to the height and width
    uh= np.random.uniform(0, .10)
    uhc = 1 - uh
    uw = np.random.uniform(.05, .2)
    uwc = 1 - uw
    lh = int(h * uh)
    hh = int(h * uhc)
    lw = int(w * uw)
    hw = int(w * uwc)

    # crop the image to new h and w
    cropped_img = img[lh:hh, lw:hw]
    hfac = np.random.randint(1, 3)
    resize_img = cv.resize(cropped_img, (w, h))

    return cropped_img


read_from_dir = './short/'
save_folder = './CROPPED/'

# loop through all the images in the folder
for dir in os.listdir(read_from_dir):
    if dir == '.DS_Store':
        continue
    else:
        currentpath = read_from_dir + dir + "/"
        altpath = dir + "/"

        # print(currentpath)
        for filename in os.listdir(currentpath):
            if filename == '.DS_Store':
                continue
            else:
                # read the image
                img = cv.imread(currentpath + filename, cv.IMREAD_UNCHANGED)

                #random brightness
                img_bright = tf.image.random_brightness(img, max_delta = 0.1, seed=None)

                #random contrast
                img_contrast = tf.image.random_contrast(img_bright, lower = 20, upper = 100, seed=None)

                #random gaussian noise
                img_noise = tf.random.normal(shape=tf.shape(img_contrast), mean=0.0, stddev=0.1, dtype=tf.float32, seed=None)

                #random flip using tensorflow
                img_flip = tf.image.random_flip_left_right(img_noise, seed=None)

                #random rotate using tensorflow
                img_rotate = tf.image.rot90(img_flip, k=1, name=None)

                #remove alpha channel
                img_rgb = rgba2rgb(img_noise)

                #random hue
                img_hue = tf.image.random_hue(img_rgb, max_delta = 0.1, seed=None)

                #random saturation
                img_sat = tf.image.random_saturation(img_hue, lower = 20, upper = 100, seed=None)

                #random image quality
                img_quality = tf.image.random_jpeg_quality(img_sat, min_jpeg_quality = 20, max_jpeg_quality = 100, seed=None)
                
                #random shift
                # Set shift values
                height_shift = tf.random.uniform([], -50, 50, dtype=tf.int32)
                width_shift = tf.random.uniform([], -50, 50, dtype=tf.int32)

                #apply shift
                img_shift = tf.roll(img_quality, shift=[height_shift, width_shift], axis=[0, 1])

                #random scaling
                # Generate random scale factor
                scale_factor = tf.random.uniform([], 0.8, 1.2)

                # Calculate new height and width
                height, width, _ = tf.unstack(tf.shape(img_shift))
                new_height = tf.cast(tf.cast(height, tf.float32) * scale_factor, tf.int32)
                new_width = tf.cast(tf.cast(width, tf.float32) * scale_factor, tf.int32)

                # Resize image
                img_scale = tf.image.resize(img_shift, [new_height, new_width])
                
                img_np = np.array(img_scale)
                img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)  # Convert from RGB to BGR color space (required by cv2.imwrite)
                img_np = img_np.astype(np.uint8)  # Convert data type to uint8 (required by cv2.imwrite)

                # crop the image
                img_crop = crop_image2(img_np)

                # save the cropped image in a new folder with the same name as the original folder in save_folder
                cv.imwrite(save_folder + altpath + filename, img_np)

    # # read the image
    # img = cv.imread(read_from_dir + filename, cv.IMREAD_UNCHANGED)

    # # crop the image
    # cropped_img = crop_image2(img)

    # # save the cropped image
    # cv.imwrite(save_folder + filename, cropped_img)


# crop_image(img, 1540, 789)