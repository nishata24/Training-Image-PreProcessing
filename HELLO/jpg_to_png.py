import PIL.Image
# Adding the GUI interface
#from tkinter import *
import os
import cv2
 

img_path = ""

save_folder = "G:\\Nishat_DLC\\MPII-human-nishat-2023-02-09\\labeled-data\MPIIpng"

# To convert the image From JPG to PNG
for dir in os.listdir(img_path):
    if dir == '.DS_Store':
        continue
    else:
        currentpath = img_path + dir + "/"
        altpath = dir + "/"
    for filename in os.listdir(img_path):
        if filename == '.DS_Store':
            continue
        else:
            img = cv2.imread(currentpath + filename, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(save_folder + altpath + filename + ".png", img)
	    