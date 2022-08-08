# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:41:16 2019

@author: Armi Tiihonen
"""
#from Color_plotting import plot_colors
from RGB_extractor import rgb_extractor
#from RGB_extractor_Xrite_CC import rgb_extractor_Xrite_CC
from Color_operations import color_calibration_results, color_conversion_results, plot_colors
from Video import save_as_video
import os
import pandas as pd
import numpy as np
#import functools as ft
#import numpy.matlib as matlib
#from matplotlib import pyplot as plt
#from colormath.color_objects import LabColor, sRGBColor
#from colormath.color_conversions import convert_color

# Purpose:
# Saves the extracted rgb or lab data.
# Input:
# results: a list that is either a direct output of rgb_extractor() or is
#          assembled as [sample, sample_percentiles_lo, sample_percentiles_hi,
#          CC, times, fig_CC, fig_samples]
# colorspace: input either 'RGB' or 'Lab'
# calibrated: input either 0 (raw data) or 1 (color calibrated data)
def save_results(results, colorspace, calibrated, sample_description):

    sample = results[0]
    sample_percentiles_lo = results[1]
    sample_percentiles_hi = results[2]
    CC = results[3]
    times = results[4]
    fig_CC = results[5]
    fig_samples = results[6]

    folderpath = ''
    if colorspace == 'RGB':
        if calibrated == 0:
            folderpath = './RGB/Raw'
            filename_body = ['_r.npy', '_g.npy', '_b.npy']
        elif calibrated == 1:
            folderpath = './RGB/Calibrated'
            filename_body = ['_r_cal.csv', '_g_cal.csv', '_b_cal.csv']
    elif colorspace == 'Lab':
        if calibrated == 0:
            folderpath = './Lab/Raw'
            filename_body = ['_Ll.csv', '_La.csv', '_Lb.csv']
        elif calibrated == 1:
            folderpath = './Lab/Calibrated'
            filename_body = ['_Ll_cal.csv', '_La_cal.csv', '_Lb_cal.csv']
            
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
            
    np.save(folderpath+'/sample'+filename_body[0], sample[0])
    np.save(folderpath+'/sample'+filename_body[1], sample[1])
    np.save(folderpath+'/sample'+filename_body[2], sample[2])
    np.savetxt(folderpath+'/CC'+filename_body[0], CC[:,:,0], delimiter=",")
    np.savetxt(folderpath+'/CC'+filename_body[1], CC[:,:,1], delimiter=",")
    np.savetxt(folderpath+'/CC'+filename_body[2], CC[:,:,2], delimiter=",")
    np.savetxt(folderpath+'/sample_percentiles_lo'+filename_body[0], sample_percentiles_lo[:,:,0], delimiter=",")
    np.savetxt(folderpath+'/sample_percentiles_lo'+filename_body[1], sample_percentiles_lo[:,:,1], delimiter=",")
    np.savetxt(folderpath+'/sample_percentiles_lo'+filename_body[2], sample_percentiles_lo[:,:,2], delimiter=",")
    np.savetxt(folderpath+'/sample_percentiles_hi'+filename_body[0], sample_percentiles_hi[:,:,0], delimiter=",")
    np.savetxt(folderpath+'/sample_percentiles_hi'+filename_body[1], sample_percentiles_hi[:,:,1], delimiter=",")
    np.savetxt(folderpath+'/sample_percentiles_hi'+filename_body[2], sample_percentiles_hi[:,:,2], delimiter=",")
    np.savetxt(folderpath+"/times.csv", times, delimiter=",")

    #fig_samples.savefig(folderpath+'/Samples.pdf')
    #fig_CC.savefig(folderpath+'/Small_CC.pdf')
    
    #Let's save the details of the samples in a format that is compatible with
    #GPyOpt_Campaign.
    sample_holder_locations = sample_description[0]
    sample_ids = sample_description[1]
    sample_compositions = sample_description[2]
    elements = sample_description[3]
    comments = sample_description[4]

    # Let's form the string that will be printed into the graphs.
    name_composition = {}
    t=''
    for i in range(0,len(sample_holder_locations)):
        t = '#' + str(i) + '-' + sample_holder_locations[i] + '-' + sample_ids[i]  + '-'
        for j in range(0, len(elements)):
            t = t + elements[j] + str(sample_compositions[i][j]) + ' '
            name_composition.update( {i : t} )
    #Compositions dataframe, manually created from array
    df_compositions = pd.DataFrame(np.array(sample_compositions), columns=elements)
    #Add sample name to dataframe of compositions
    df_compositions.insert(loc=0, column='Sample', value=pd.Series(name_composition))
    #Add comment field to dataframe of compositions
    df_compositions.insert(loc=0, column='Comments', value=pd.Series(comments))
    
    df_compositions.to_csv(folderpath+'/Samples.csv')
    
    return (None)

###############################################################################
# The locations at the sample holder are presented in this order in the code.
# The same order holds through all the code.
sample_holder_locations = ['D1', 'C1', 'B1', 'A1',
                           'D2', 'C2', 'B2', 'A2',
                           'D3', 'C3', 'B3', 'A3',
                           'D4', 'C4', 'B4', 'A4',
                           'D5', 'C5', 'B5', 'A5',
                           'D6', 'C6', 'B6', 'A6',
                           'D7', 'C7', 'B7', 'A7']
#######################################################################

# FILL IN THE DETAILS OF THE AGING TEST:

# Fill in the sample ids in string format in the same order than is presented
# in variable 'sample_holder_locations' (see above). If the position has been
# left empty, give '-'.
sample_ids = ['0', '0', '0', '0',
              '0', '0', '0', '0',
              '0', '0', '0', '0',
              '0', '0', '0', '0',
              '0', '0', '0', '0',
              '0', '0', '0', '0',
              '0', '0', '0', '0']

# Which elements you are optimizing in the sample composition? E.g.,
# elements = ['Cs', 'FA', 'MA']
elements = ['MAPbI', 'FAPbI', 'CsPbI']
# Fill in sample compositions. Give zero values if the position is
# empty. Otherwise the compositions should sum up till 100. E.g.,
# 10% Cs, 50% MA, 40% FA would be written as [0.1, 0.5, 0.4].
sample_compositions = np.array(
        [[1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00],
        [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00],
        [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00],
        [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00],
        [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00],
        [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00],
        [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00], [1.00,0.00,0.00]])

# Do you have any other free-form comments about the samples? Give '-' if you
# don't have any.
comments = ['-', '-', '-', '-',
            '-', '-', '-', '-',
            '-', '-', '-', '-',
            '-', '-', '-', '-',
            '-', '-', '-', '-',
            '-', '-', '-', '-',
            '-', '-', '-', '-']

# Give the path to the folder that contains the pictures (without ending
# slash). Use '/' for linux and '\' for Windows.
#
# Important: You need to clean up the data before running the code. The code
# assumes the (alphabetically) first picture in the folder has been taken of
# Xrite color chart (i.e., this picture will not be analyzed) and all the other
# pictures are taken of sample holder (these pictures will be analyzed). 
pic_folder = 'C:\\Users\\tomim\OneDrive\Työpöytä\Työt\Python\Example data and code - RGB\Data\\20201112-R1-RN\BMP'
# Give the name of the picture that has been taken taken of Xrite color chart.
pic_name_Xrite = '20201112223424.bmp'

# Give the settings for finding the samples and color chart patches from the
# pics.
# Optimize using Color_operations.py if necessary. Explanations in the same ile.
crop_box_CC = (483,200,680,320) # Small color chart
offset_array_CC = [[8,8],[8,8]]
crop_box_samples = (270,390,785,845) # Films on sample holder
offset_array_samples = [[33,18],[18,18]]
crop_box_Xrite = (380,250+240,830,790) # Xrite passport
offset_array_Xrite = [[20,20],[20,20]]

# How often do you want to print out figures? The cropping of every nth pic
# will be printed to the console. For short aging tests, value 1 is
# good, for very long ones you might put even 250 to make the code run faster.
print_out_interval = 3

###############################################################################

# For Campaign 2.0, we assume the sum of the compositions to be exactly 1 (100%).
# This does not necessarily hold for other projects. This property is meant for
# ensuring integrity in cases where the sum of sample materials sums to 0.99
# instead of 1 because of rounding.
for i in range(0,len(sample_compositions)):
    if (np.sum(sample_compositions[i]) != 1) & (np.sum(sample_compositions[i]) != 0):
        sample_compositions[i] = np.round(sample_compositions[i]/np.sum(
                sample_compositions[i]),2)
        

sample_description = [sample_holder_locations, sample_ids, sample_compositions,
                      elements, comments]








# THE RESULTS in rgb color space (no color calibration)
results_rgb = rgb_extractor(pic_folder, crop_box_samples, offset_array_samples,
                            crop_box_CC, offset_array_CC, print_out_interval)
save_results(results_rgb, 'RGB', 0, sample_description)
# Output explained: 
# results_rgb = [sample_rgb, sample_rgb_percentiles_lo,
# sample_rgb_percentiles_hi, CC_rgb, times, fig_CC_rgb, fig_samples_rgb]
# - sample_rgb[samples 0...27][times 0...][0:R/1:G/2:B]: rgb values of each
#   sample at each moment
# - sample_rgb_percentiles_lo: lower percentiles of each sample at each moment,
#   same format as above 
# - sample_rgb_percentiles_hi: higher percentiles of each sample at each moment,
#   same format as above 
# - CC_rgb[color patches 0...23][times 0...][0:R/1:G/2:B]: rgb values of each
#   color patch in the small reference color chart at each moment
# - times[times 0...]: each sampling moment (minutes after the beginning of the
#   aging test; that is the time defined in the filename of the first picture)
# - fig_CC_rgb: a plot about rgb values vs time in each color patch of the
#   small color chart
# - fig_samples_rgb: a plot about rgb values vs time in each sample
# - picfiles: filenames in the same order than the data is

# Let's convert these to Lab (no color calibration)
results_lab = color_conversion_results(results_rgb)
save_results(results_lab, 'Lab', 0, sample_description)   

# Let's perform color calibration and save the data in both RGB and Lab
# formats.
[results_rgb_cal, results_lab_cal] = color_calibration_results(results_rgb, results_lab,
    [pic_folder, pic_name_Xrite, crop_box_Xrite, offset_array_Xrite])    
save_results(results_rgb_cal, 'RGB', 1, sample_description)   
save_results(results_lab_cal, 'Lab', 1, sample_description)   

# Let's produce a video about the pictures (i.e. corresponds to raw RGB data).
# This should be commented unless you have the necessary software installed.
save_as_video(pic_folder, crop_box_samples, crop_box_CC, 1, 'bmp')


# TO DO: This part does not necessarily work for Windows at the moment. Need to
# be updated. Until that, keep commented.
# 
# Let's plot images illustrating the selected sample areas from the pictures
# and the determined color of each sample. This is done for both unedited
# pictures (i.e., raw RGB) and color calibrated data (i.e., calibrated RGB).
# These pictures can be utilized for checking if the selected areas are correct
# and for checking how color calibration affects the colors.
# Additionally, we make a video on both folders.
crop_box = crop_box_samples
offset_array = offset_array_samples
save_to_folder_raw = './RGB/Raw/Cropped pics'
save_to_folder_cal = './RGB/Calibrated/Cropped pics'
if not os.path.exists(save_to_folder_raw):
        os.makedirs(save_to_folder_raw)
if not os.path.exists(save_to_folder_cal):
        os.makedirs(save_to_folder_cal)
savefig = 1
print_out = 0
for i in range(0,len(results_rgb[-1])):
    if (print_out_interval > 0) & (i % print_out_interval == 0):
        print_out = 1
        pic_path = results_rgb[-1][i]
        color_array_raw = results_rgb[0][:,:,:,:,i]
        color_array_cal = results_rgb_cal[0][:,:,:,:,i]
        #print('i',i)
        plot_colors(pic_path, crop_box, offset_array,
                color_array_raw, save_to_folder_raw, savefig, '', print_out)
        plot_colors(pic_path, crop_box, offset_array,
                color_array_cal, save_to_folder_cal, savefig, '', print_out)
    else:
        print_out = 0
    
# These lines should be commented unless you have the necessary software installed.
save_as_video(save_to_folder_raw, crop_box_samples, crop_box_CC, 0, 'jpg', 'Raw_mean_colors')
save_as_video(save_to_folder_cal, crop_box_samples, crop_box_CC, 0, 'jpg', 'Calibrated_mean_colors')


print("end")