#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:52:22 2019

@author: IsaacParker
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from os import walk
from os import listdir
from os.path import isfile, join



def get_image(image_path, crop_box):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    image = image.crop(box=crop_box) #box=(left, upper, right, lower)
    [width,height] = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((height, width, channels))
    image.close()
    return (width,height,pixel_values)

def image_slicing(image_array, col_num,row_num,offset_array):
    """slice the ROIs from an image of an array of samples/colorcard"""
    row_h = int(image_array.shape[0]/row_num)
    col_w = int(image_array.shape[1]/col_num)
    
    fig,ax = plt.subplots(1,figsize=(5,5))
    images = []
    imagecol = []
    for y in np.arange(row_num):
        imagerow = []
        for x in np.arange(col_num):
            # slicing indices for each color square
            y1 = row_h*y+offset_array[1][0]
            y2 = row_h*(y+1)-offset_array[1][1]
            x1 = col_w*x+offset_array[0][0]
            x2 = col_w*(x+1)-offset_array[0][1]
            image = image_array[y1:y2,x1:x2]
            imagerow.append(image)#append every images in a row into a row list
            images.append(image)#append every images into a list
            # Add the rectangular patch to the Axes
            # Create a Rectangle patch
            lw=1 # Line width
            ec='r' # edge color
            fc='none' # face color
            rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                      linewidth=lw,edgecolor=ec,facecolor=fc)
            ax.add_patch(rect)
    
        imagecol.append(np.concatenate(imagerow, axis=1))
    image_reconstr = np.array(np.concatenate(imagecol, axis=0), dtype=np.uint8)
    return [fig, ax, image_reconstr, images]



# Purpose:
# Extracts the RGB colors of the samples in all pictures in the folder (except for
# the first picture that is of Xrite color chart).
# 
# Input:
# - pic_folder: folder path as a string without a slash in the end.
# - pic_name: picture file name as a string.
# - crop_box_samples: a tuple of integers defining the (left, upper, right, lower)
#   borders of the crop box in pixels. Use function test_crop_box in
#   Test_crop_box.py for checking which values are good.
# - offset_array_samples: a list defining the amount of pixels to be discarded from
#   the [[left,right],[upper,lower]] side of each item. Use function
#   test_crop_box in Test_crop_box.py for checking which values are good.
# - crop_box_CC: Crop box for the small color chart visible in all the pictures.
# - offset_array_CC: Offset array for the small color chart.
# - print_out_interval: an integer defining how often this code prints figures.
#    
# Output:
# - results: a list collecting the results of this code. Contents: [sample_rgb,
#   sample_rgb_percentiles_lo, sample_rgb_percentiles_hi,CC_rgb, times,
#   fig_CC_rgb, fig_samples_rgb]. Contents explained:
#   - sample_rgb[samples 0...27][times 0...][0:R/1:G/2:B]: rgb values of each
#     sample at each moment (a NumPy array)
#   - sample_rgb_percentiles_lo: lower percentiles of each sample at each moment,
#     same format as above (a NumPy array)
#   - sample_rgb_percentiles_hi: higher percentiles of each sample at each moment,
#     same format as above (a NumPy array)
#   - CC_rgb[color patches 0...23][times 0...][0:R/1:G/2:B]: rgb values of each
#     color patch in the small reference color chart at each moment (a NumPy array)
#   - times[times 0...]: each sampling moment (minutes after the beginning of the
#     aging test; that is the time defined in the filename of the first picture)
#     (a NumPy array)
#   - fig_CC_rgb: a plot about rgb values vs time in each color patch of the
#     small color chart
#   - fig_samples_rgb: a plot about rgb values vs time in each sample
def rgb_extractor(pic_folder, crop_box_samples, offset_array_samples,
                  crop_box_CC, offset_array_CC, print_out_interval):
    #%%
    # Read all the files from a specified directory
    files=[]
    for (dirpath, dirnames, filenames) in walk(pic_folder):
        for filename in filenames:
            if filename!='.DS_Store':
                files.append(dirpath+'/'+filename)
    files.sort()
    
    #%%
    # test picture for adjusting the cropping box
    # should try mutiple ones for the optimium cropping box options
    testfile = files[1]
    print(testfile)
     
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
    
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    # Croping Box for Sample Region (TO BE CHANGED)
    crop_box_Sample = crop_box_samples#(350+35,250+60,900-20,850-75)#(left, upper, right, lower)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    box= crop_box_Sample
    rect1 = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                              linewidth=lw,edgecolor=ec,facecolor=fc)
    
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    # Croping Box for Color Card (TO BE CHANGED)
    crop_box_ColorCard = crop_box_CC#(571,111,770,245)#(left, upper, right, lower)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    box= crop_box_ColorCard
    rect2 = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                              linewidth=lw,edgecolor=ec,facecolor=fc)
    # Add the patch to the Axes
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    plt.show()
    
    # Color Card Cropping
    [w,h,image_ColorCard]=get_image(testfile,crop_box_ColorCard)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    # Row, Columns Settings and Offset pixels for each color square (TO BE CHANGED)
    row_num_CC=4
    col_num_CC=6
    #offset_array_CC = [[6,5],[16,5]]#[[x_left,x_right],[y_upper,y_lower]]
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    [fig_CC, ax_CC, reconstr_CC, image_CC]= image_slicing(
            image_ColorCard, col_num_CC, row_num_CC, offset_array_CC)
    
    ax_CC.imshow(Image.fromarray(np.array(image_ColorCard, dtype=np.uint8), 'RGB'))
    plt.show()
    
    fig,ax = plt.subplots(1,figsize=(5,5))
    ax.imshow(Image.fromarray(reconstr_CC, 'RGB'))
    plt.show()
    
    # Sample Cropping
    [w,h,image_ROI_Sample]=get_image(testfile,crop_box_Sample)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    # Row, Columns Settings and Offset pixels for each sample (TO BE CHANGED)
    row_num_Sample=7
    col_num_Sample=4
    offset_array_Sample = offset_array_samples#[[40,10],[15,10]]#[[x_left,x_right],[y_upper,y_lower]]
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    [fig_ROI, ax_ROI, reconstr_ROI, image_ROI]= image_slicing(
            image_ROI_Sample, col_num_Sample, row_num_Sample, offset_array_Sample)
    
    ax_ROI.imshow(Image.fromarray(np.array(image_ROI_Sample, dtype=np.uint8), 'RGB'))
    plt.show()
    
    fig,ax = plt.subplots(1,figsize=(5,5))
    ax.imshow(Image.fromarray(reconstr_ROI, 'RGB'))
    plt.show()
    
    #%% extract RGB from all the image in time series
    CC_r_timeseries = []
    CC_g_timeseries = []
    CC_b_timeseries = []
    
    sample_r_timeseries = []
    sample_g_timeseries = []
    sample_b_timeseries = []
    
    sample_r_hi_timeseries = []
    sample_g_hi_timeseries = []
    sample_b_hi_timeseries = []
    
    sample_r_lo_timeseries = []
    sample_g_lo_timeseries = []
    sample_b_lo_timeseries = []
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    picfiles = files[1::] #Select the use fully pictures only. Skips the first
    # picture (i.e., Xrite color passport picture).
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    time_fmt = '%Y%m%d%H%M%S'# grabbing the time from the filenames
    prefix_len = len(dirpath)+1
    time0 = datetime.strptime(picfiles[0][prefix_len:-4], time_fmt)
    t = []
    counter = 0
    for file in picfiles:
        counter = counter +1
        if (print_out_interval > 0) & (counter % print_out_interval == 0):
            print(counter)
            print(file)
        time = datetime.strptime(file[prefix_len:-4], time_fmt) 
        t.append((time - time0).total_seconds()/60)
     
        [w,h,image_ColorCard]=get_image(file,crop_box_ColorCard)
        [fig_CC, ax_CC, reconstr_CC, image_CC]= image_slicing(image_ColorCard, col_num_CC, row_num_CC, offset_array_CC)
        if (print_out_interval > 0) & (counter % print_out_interval == 0):
            ax_CC.imshow(Image.fromarray(np.array(image_ColorCard, dtype=np.uint8), 'RGB'))
            plt.show()
        plt.close(fig_CC)
        
        CC_RGBlist = []
        for img in image_CC:
            CC_RGBlist.append([np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])])
        CC_r_timeseries.append(np.array(CC_RGBlist)[...,0])
        CC_g_timeseries.append(np.array(CC_RGBlist)[...,1])
        CC_b_timeseries.append(np.array(CC_RGBlist)[...,2])
        
        
        [w,h,image_ROI_Sample]=get_image(file,crop_box_Sample)
        [fig_ROI, ax_ROI, reconstr_ROI, image_ROI]= image_slicing(
            image_ROI_Sample, col_num_Sample, row_num_Sample, offset_array_Sample)
        if (print_out_interval > 0) & (counter % print_out_interval == 0):
            ax_ROI.imshow(Image.fromarray(np.array(image_ROI_Sample, dtype=np.uint8), 'RGB'))
            plt.show()
        plt.close(fig_ROI)
        
        
        Sample_RGBlist = []
        Sample_RGBlist_hi = []
        Sample_RGBlist_lo = []
        for img in image_ROI:
            [r,g,b] = [np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])]    #mean here!!!
            [r_hi,g_hi,b_hi] = [np.percentile(img[:,:,0],95),np.percentile(img[:,:,1],95),np.percentile(img[:,:,2],95)]
            [r_lo,g_lo,b_lo] = [np.percentile(img[:,:,0],5),np.percentile(img[:,:,1],5),np.percentile(img[:,:,2],5)]
    
            Sample_RGBlist.append([r,g,b])
            Sample_RGBlist_hi.append([r_hi,g_hi,b_hi])
            Sample_RGBlist_lo.append([r_lo,g_lo,b_lo])
        sample_r_timeseries.append(np.array(Sample_RGBlist)[...,0])
        sample_g_timeseries.append(np.array(Sample_RGBlist)[...,1])
        sample_b_timeseries.append(np.array(Sample_RGBlist)[...,2])
        
        sample_r_hi_timeseries.append(np.array(Sample_RGBlist_hi)[...,0])
        sample_g_hi_timeseries.append(np.array(Sample_RGBlist_hi)[...,1])
        sample_b_hi_timeseries.append(np.array(Sample_RGBlist_hi)[...,2])
        sample_r_lo_timeseries.append(np.array(Sample_RGBlist_lo)[...,0])
        sample_g_lo_timeseries.append(np.array(Sample_RGBlist_lo)[...,1])
        sample_b_lo_timeseries.append(np.array(Sample_RGBlist_lo)[...,2])
    
    
        
        
    
    # change the time relative to the time0 (in case it hasn't)   
    t=np.array([i-min(t) for i in t])
    # RGB time series for Color Card
    CC_r_timeseries = np.array(CC_r_timeseries).T
    CC_g_timeseries = np.array(CC_g_timeseries).T
    CC_b_timeseries = np.array(CC_b_timeseries).T
    
    # RGB time series for Sample
    sample_r_timeseries = np.array(sample_r_timeseries).T
    sample_g_timeseries = np.array(sample_g_timeseries).T
    sample_b_timeseries = np.array(sample_b_timeseries).T
    
    sample_r_hi_timeseries = np.array(sample_r_hi_timeseries).T
    sample_g_hi_timeseries = np.array(sample_g_hi_timeseries).T
    sample_b_hi_timeseries = np.array(sample_b_hi_timeseries).T
    sample_r_lo_timeseries = np.array(sample_r_lo_timeseries).T
    sample_g_lo_timeseries = np.array(sample_g_lo_timeseries).T
    sample_b_lo_timeseries = np.array(sample_b_lo_timeseries).T
    
    # Sorting and return values
    #x = np.array([6, 7, 1, 2])
    #a=np.array([6, 7, 1, 2])
    #b=np.array([2, 1, 6, 7])
    #c=np.array([7, 6, 2, 1])
    #order = np.argsort(x)
    #a = x[order]
    #Mnew = np.array([b[order],c[order],d[order]])
    #Morig = np.sort(np.array([x,a,b,c]),axis = 1)
    #print(Mnew)
    #print(Morig)
    
    order = np.argsort(t)
    t_sort = t[order]
    r_sort = sample_r_timeseries[:,order]
    g_sort = sample_g_timeseries[:,order]
    b_sort = sample_b_timeseries[:,order]
    CC_r_sort = CC_r_timeseries[:,order]
    CC_g_sort = CC_g_timeseries[:,order]
    CC_b_sort = CC_b_timeseries[:,order]
    r_hi_sort = sample_r_hi_timeseries[:,order]
    g_hi_sort = sample_g_hi_timeseries[:,order]
    b_hi_sort = sample_b_hi_timeseries[:,order]
    r_lo_sort = sample_r_lo_timeseries[:,order]
    g_lo_sort = sample_g_lo_timeseries[:,order]
    b_lo_sort = sample_b_lo_timeseries[:,order]
    picfiles_sort = list( picfiles[i] for i in order)
    
    sample_rgb = [r_sort, g_sort, b_sort]
    sample_rgb_percentiles_lo = [r_lo_sort, g_lo_sort, b_lo_sort] 
    sample_rgb_percentiles_hi = [r_hi_sort, g_hi_sort, b_hi_sort]
    CC_rgb = [CC_r_sort, CC_g_sort, CC_b_sort]
    times = t_sort
    

    
    
    #%% RGB vs time plots (sorting disabled because has already been done above)
    [fig_CC, axs_CC, fig_samples, axs_samples] = plot_aging_data(
                    row_num_CC,col_num_CC,t_sort, CC_r_sort, CC_g_sort, CC_b_sort,
                    row_num_Sample,col_num_Sample,r_sort, g_sort, b_sort,
                    r_hi_sort, g_hi_sort, b_hi_sort,
                    r_lo_sort, g_lo_sort, b_lo_sort, 'RGB')
    
    results = [sample_rgb, sample_rgb_percentiles_lo, sample_rgb_percentiles_hi, CC_rgb]
    
    counter = 0
    for result_item in results:
        result_item = np.array(result_item)
        result_item = np.swapaxes(result_item,0,1)
        result_item = np.swapaxes(result_item,1,2)
        results[counter] = result_item
        counter = counter + 1
    results.extend([np.array(times), fig_CC, fig_samples, picfiles_sort])
    
    return results
    












def plot_aging_data(row_num_CC,col_num_CC,t_sort, CC_r_sort, CC_g_sort, CC_b_sort,
                    row_num_Sample,col_num_Sample,r_sort, g_sort, b_sort,
                    r_hi_sort, g_hi_sort, b_hi_sort,
                    r_lo_sort, g_lo_sort, b_lo_sort, datatype):


    fig_CC, axs_CC = plt.subplots(row_num_CC,col_num_CC,figsize=(10,6),sharex=True,sharey=True)
    for r in np.arange(row_num_CC):
        for c in np.arange(col_num_CC):
            i = c + r*col_num_CC
            #[t_sort, CC_r_sort,CC_g_sort,CC_b_sort] = np.sort(np.array([t,
            #         CC_r_timeseries[i],
            #         CC_g_timeseries[i],
            #         CC_b_timeseries[i]]),axis = 1)
        
            axs_CC[r][c].plot(t_sort, CC_r_sort[i],color='r')
            axs_CC[r][c].plot(t_sort, CC_g_sort[i],color='g')
            axs_CC[r][c].plot(t_sort, CC_b_sort[i],color='b')    
            
    if datatype == 'RGB':
        fig_CC.text(0.5, 0.90,'ColorCard RGB (rows - columns)',ha='center',fontsize=14)
        fig_CC.text(0.06, 0.5,'RGB Value [0-256]',va='center',rotation='vertical',fontsize=12)
    elif datatype == 'Lab':
        fig_CC.text(0.5, 0.90,'ColorCard Lab (rows - columns)',ha='center',fontsize=14)
        fig_CC.text(0.06, 0.5,'Lab Value [-100 - +100]',va='center',rotation='vertical',fontsize=12)
    fig_CC.text(0.5, 0.04,'Time [min]',ha='center', fontsize=12)
    plt.show()
    
    
    
    fig_samples, axs_samples = plt.subplots(row_num_Sample,col_num_Sample,figsize=(10,10),sharex=True,sharey=True)
    for r in np.arange(row_num_Sample):
        for c in np.arange(col_num_Sample):
            i = c + r*col_num_Sample
            """[t_sort, r_sort,g_sort,b_sort] = np.sort(np.array([t,
                     sample_r_timeseries[i],
                     sample_g_timeseries[i],
                     sample_b_timeseries[i]]),axis = 1)           
            [t_sort, r_hi_sort,g_hi_sort,b_hi_sort] = np.sort(np.array([t,
                     sample_r_hi_timeseries[i],
                     sample_g_hi_timeseries[i],
                     sample_b_hi_timeseries[i]]),axis = 1)
            [t_sort, r_lo_sort,g_lo_sort,b_lo_sort] = np.sort(np.array([t,
                     sample_r_lo_timeseries[i],
                     sample_g_lo_timeseries[i],
                     sample_b_lo_timeseries[i]]),axis = 1)"""
            
            axs_samples[r][c].plot(t_sort, r_sort[i],color='r')      
            axs_samples[r][c].plot(t_sort, g_sort[i],color='g')
            axs_samples[r][c].plot(t_sort, b_sort[i],color='b')
            axs_samples[r][c].plot(t_sort, r_hi_sort[i],'--',color='r')
            axs_samples[r][c].plot(t_sort, g_hi_sort[i],'--',color='g')
            axs_samples[r][c].plot(t_sort, b_hi_sort[i],'--',color='b')       
            axs_samples[r][c].plot(t_sort, r_lo_sort[i],':',color='r')
            axs_samples[r][c].plot(t_sort, g_lo_sort[i],':',color='g')
            axs_samples[r][c].plot(t_sort, b_lo_sort[i],':',color='b') 
            if datatype == 'RGB':
                plt.ylim((0, 150))
            elif datatype == 'Lab':
                plt.ylim((-100, 100))
    #
    if datatype == 'RGB':
        fig_samples.text(0.5, 0.90,'Sample RGB (rows - columns) [dotted: 5%perc; dashed: 95%perc; solid: mean]',ha='center',fontsize=14)
        fig_samples.text(0.06, 0.5,'RGB Value [0-256]',va='center',rotation='vertical',fontsize=12)
    elif datatype == 'Lab':
        fig_samples.text(0.5, 0.90,'Sample Lab (rows - columns) [dotted: 5%perc; dashed: 95%perc; solid: mean]',ha='center',fontsize=14)
        fig_samples.text(0.06, 0.5,'Lab Value [-100 - +100]',va='center',rotation='vertical',fontsize=12)
    fig_samples.text(0.5, 0.08,'Time [min]',ha='center',fontsize=12)
    
    plt.show()
    
    return [fig_CC, axs_CC, fig_samples, axs_samples]