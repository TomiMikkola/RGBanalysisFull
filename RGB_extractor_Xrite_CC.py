#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:52:22 2019

@author: IsaacParker, Liu Zhe, Armi Tiihonen
"""

from RGB_extractor import get_image, image_slicing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Purpose:
# Extracts the colors in the color patches of Xrite color chart. Does the same
# than RGB_extractor but does not step through multiple files and returns a np
# array with dimensions (24,3) instead of (24, N_times, 3).
# 
# Input:
# - pic_folder: folder path as a string without a slash in the end.
# - pic_name: picture file name as a string.
# - crop_box: a tuple of integers defining the (left, upper, right, lower)
#   borders of the crop box in pixels. Use function test_crop_box in
#   Test_crop_box.py for checking which values are good.
# - offset_array: a list defining the amount of pixels to be discarded from
#   the [[left,right],[upper,lower]] side of each item. Use function
#   test_crop_box in Test_crop_box.py for checking which values are good.
#
# Output:
# - Xrite_rgb: A NumPy array defining the mean of the colors in the color patches
#   of Xrite color chart. Dimensions: (N_color_patches=24, N_color_channels=3).
def rgb_extractor_Xrite_CC(pic_folder, pic_name, crop_box, offset_array):

    #%%
    # fetch picture for adjusting the cropping box
    # should try find the optimum cropping box options
    testfile = pic_folder+'/'+pic_name
     
    image = Image.open(testfile, 'r')
    im = Image.fromarray(np.array(image, dtype=np.uint8), 'RGB')
    # Create figure and axes
    fig,ax = plt.subplots(1,figsize=(1248/100,1024/100))
    # Display the image
    ax.imshow(im)
    # Create a Rectangle patch
    lw=1 # Line width
    ec='r' # edge color
    fc='none' # face color

    box= crop_box
    rect2 = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                              linewidth=lw,edgecolor=ec,facecolor=fc)
    # Add the patch to the Axes
    ax.add_patch(rect2)
    plt.show()
    
    # Color Card Cropping
    [w,h,image_ColorCard]=get_image(testfile,crop_box)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    # Row, Columns Settings and Offset pixels for each color square (TO BE CHANGED)
    row_num_CC=4
    col_num_CC=6
    offset_array_CC = offset_array#[[25,25],[25,25]]#[[x_left,x_right],[y_upper,y_lower]]
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    [fig_CC, ax_CC, reconstr_CC, image_CC]= image_slicing(
            image_ColorCard, col_num_CC, row_num_CC, offset_array_CC)
    
    ax_CC.imshow(Image.fromarray(np.array(image_ColorCard, dtype=np.uint8), 'RGB'))
    plt.show()
    
    fig,ax = plt.subplots(1,figsize=(5,5))
    ax.imshow(Image.fromarray(reconstr_CC, 'RGB'))
    plt.show()
    
    #%% extract RGB from the image from 24 patches
    
    print(testfile)
        
    Xrite_rgb = []
    for img in image_CC:
        Xrite_rgb.append([np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])])
    Xrite_rgb = np.array(Xrite_rgb)
    
    return (Xrite_rgb)


#test=rgb_extractor_Xrite_CC('.', '20190328112533_40_np.jpg', (350+60,250+245,900-80,850-80), [[20,20],[20,20]])