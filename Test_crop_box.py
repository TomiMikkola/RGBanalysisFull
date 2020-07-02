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


# This function is used for plotting the given crop box for the given picture.
# Input:
# pic_folder: folder path as a string without ending slash.
# pic_name: picture file name as a string.
# crop_box: a tuple of integers defining the (left, upper, right, lower)
# borders of the crop box in pixels. (350+60,250+245,900-80,850-80) is a good
# initial guess for a color chart, and (350+5,250+110,900-50,850-35) for samples.
# offset_array: A list defining the area that is analyzed for each sample or
# color patch of a color chart. Crop box is sliced to as many parts as there
# items to be analyzed in the picture. After that, the edge area is discarded
# the analysis. This input defines the amount of pixels to be discarded from
# the [[left,right],[upper,lower]] side of each item. [[20,20],[20,20]] is a
# good initial guess for a color chart, and [[25,15],[12,12]] is for samples.
# picture_target: 0 (if picture is from a color chart) or 1 (if picture is from
# a sample holder)
# 
# Output:
# Nothing.

def test_crop_box(pic_folder, pic_name, crop_box, offset_array, picture_target):

    # If the picture is from a color chart:
    if picture_target == 0:
        row_num=4
        col_num=6
    elif picture_target == 1:
        row_num=7
        col_num=4
    else:
        raise Exception('Incorrect format of input!')
        return None


    #%%
    # fetch picture for adjusting the cropping box
    # should try find the optimium cropping box options
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
    
    # Cropping
    [w,h,image1]=get_image(testfile,crop_box)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    [fig, ax, reconstr, image2]= image_slicing(image1, col_num, row_num, offset_array)
    
    ax.imshow(Image.fromarray(np.array(image1, dtype=np.uint8), 'RGB'))
    plt.show()
    
    #fig,ax = plt.subplots(1,figsize=(5,5))
    #ax.imshow(Image.fromarray(reconstr, 'RGB'))
    #plt.show()
    
    print(testfile)
        
    return None

def visualize_colors(data, space, picture_target):

    if picture_target == 0: # Color chart
        row_num=4
        col_num=6
    elif picture_target == 1: # Samples
        row_num=7
        col_num=4
    else:
        raise Exception('Incorrect format of input!')
        return None

    if (space != 'Lab') and (space != 'RGB'):
        raise Exception('Incorrect format of input!')

    # We need the code to work for inputs containing the optional dimension
    # n_times (i.e., many time points) and for inputs containing only one time
    # point.
    n_d = data.ndim
    if n_d == 2:
        data = np.expand_dims(data, 1)
    elif n_d != 3:
        raise Exception('Faulty number of dimensions in the input!')
    
    #%%
    # fetch picture for adjusting the cropping box
    # should try find the optimium cropping box options
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
    
    # Cropping
    [w,h,image1]=get_image(testfile,crop_box)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    [fig, ax, reconstr, image2]= image_slicing(image1, col_num, row_num, offset_array)
    
    ax.imshow(Image.fromarray(np.array(image1, dtype=np.uint8), 'RGB'))
    plt.show()
    
    #fig,ax = plt.subplots(1,figsize=(5,5))
    #ax.imshow(Image.fromarray(reconstr, 'RGB'))
    #plt.show()
    
    print(testfile)
        
    return None

#Xrite passport    
test_crop_box('./20190423-R1-AT/BMP', '20190422121726.bmp', (360+0,280+245,850-70,850-50), [[20,20],[20,20]], 0) #476, 315
# samples
test_crop_box('./20190423-R1-AT/BMP', '20190423113654.bmp',
             (285,382,785,845), [[36,38],[23,21]], 1)
# small color chart
test_crop_box('./20190423-R1-AT/BMP', '20190423113654.bmp',
             (490,200,680,320), [[8,8],[8,8]], 0)

#
'''
crop_box_CC = (483,200,680,320) # Small color chart
offset_array_CC = [[8,8],[8,8]]
crop_box_samples = (270,390,785,845) # Films on sample holder
offset_array_samples = [[33,18],[18,18]]
crop_box_Xrite = (350+60,250+245,900-80,850-80) # Xrite passport
offset_array_Xrite = [[20,20],[20,20]]

'''
