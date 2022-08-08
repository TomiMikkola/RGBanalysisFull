#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:33:16 2019

@author: armi
"""
from RGB_extractor_Xrite_CC import rgb_extractor_Xrite_CC
from RGB_extractor import plot_aging_data, get_image, image_slicing
import numpy as np
import numpy.matlib as matlib
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from Video import substring_indexes
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import os
import requests
import concurrent.futures
from time import time




###############################################################################
# Part 1/2: Functions related to color space conversions and color calibration.
###############################################################################

# Input:
# - data: a np array with dimensions (n_samples, {optional
#   dimension: n_times}, n_color_coordinates=3) (e.g., a direct output of
#   'rgb_extractor()' or 'rgb_extractor_Xrite_CC()')
# - from_space: choose either 'RGB' or 'Lab'
# - to_space: choose either 'RGB' or 'Lab'
# Output:
# - converted: a np array with the same dimensions than in the input
def convert_color_space(data, from_space, to_space):
    # We need the code to work for inputs containing the optional dimension
    # n_times (i.e., many time points) and for inputs containing only one time
    # point.
    n_d = data.ndim
    converted = np.array([])
    if n_d == 2:
        data = np.expand_dims(data, 1)
    elif n_d == 5:
        data = data
    elif n_d != 3:
        raise Exception('Faulty number of dimensions in the input!')
    if (from_space == 'RGB') and (to_space == 'Lab'):
        # Values from rgb_extractor() are [0,255] so let's normalize.
        data = data/255
        # Transform to color objects (either sRGBColor or LabColor).
        if n_d == 5:
            print("RGB to Lab")
            data_r = data[0,:,:,:,:]
            data_g = data[1,:,:,:,:]
            data_b = data[2,:,:,:,:]
            converted = np.zeros(data.shape)
            #converted = np.array([])
            #converted2 = np.array([])

            def convert_rgb_lab_section(i):
                for j in range(data_r.shape[1]):
                    data_objects = np.vectorize(lambda x,y,z: sRGBColor(x,y,z))(
                        data_r[i,j,:,:], data_g[i,j,:,:], data_b[i,j,:,:])
                    color_space = matlib.repmat(LabColor, *data_objects.shape)
                    # Transform from original space to new space 
                    converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
                        data_objects, color_space)
                    # We got a matrix of color objects. Let's transform to a 3D matrix of floats.
                    converted3 = np.transpose(np.vectorize(lambda x: (x.lab_l, x.lab_a, x.lab_b))(
                        converted_objects), (1,2,0))
                    converted3 = np.swapaxes(converted3,0,2)
                    converted3 = np.swapaxes(converted3,1,2)
                    converted[:,i,j,:,:] = converted3

            start = time()
            processes = []
            #with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(data_r.shape[0]):
                print(f"{i}/{data_r.shape[0]}")
                #processes.append(executor.submit(convert_rgb_lab_section, i))
                convert_rgb_lab_section(i)


            #for task in as_completed(processes):
            #    print(task.result())
            #    
            #print(f'Time taken: {time() - start}')
        
        else:
            # Transform to color objects (either sRGBColor or LabColor).
            data_objects = np.vectorize(lambda x,y,z: sRGBColor(x,y,z))(
                data[:,:,0], data[:,:,1], data[:,:,2])
            # Target color space
            color_space = matlib.repmat(LabColor, *data_objects.shape)
            # Transform from original space to new space 
            converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
                data_objects, color_space)
            # We got a matrix of color objects. Let's transform to a 3D matrix of floats.
            converted = np.transpose(np.vectorize(lambda x: (x.lab_l, x.lab_a, x.lab_b))(
                converted_objects), (1,2,0))
            # This one works too. I don't know which one would be better.
            #converted = np.array([[(converted_objects[x,y].lab_l,
            #                         converted_objects[x,y].lab_a,
            #                         converted_objects[x,y].lab_b) for y in range(
            #                         converted_objects.shape[1])] for x in range(
            #                         converted_objects.shape[0])])
            # We want output to be in the same shape than the input.
    elif (from_space == 'Lab') and (to_space == 'RGB'):
        if n_d == 3:
            data_objects = np.vectorize(lambda x,y,z: LabColor(x,y,z))(
                data[:,:,0], data[:,:,1], data[:,:,2])
            color_space = matlib.repmat(sRGBColor, *data_objects.shape)
            converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
                data_objects, color_space)
            converted = np.transpose(np.vectorize(lambda x: (x.rgb_r, x.rgb_g, x.rgb_b))(
                converted_objects), (1,2,0))
            # Colormath library interprets rgb in [0,1] and we want [0,255] so let's
            # normalize to [0,255].
            converted = converted*255
        elif n_d == 5:
            print("Lab to RGB")
            converted = np.zeros(data.shape)
            for i in range(data.shape[1]):
                print(f"{i}/{data.shape[1]}")
                for j in range(data.shape[2]):
                    data_objects = np.vectorize(lambda x,y,z: LabColor(x,y,z))(
                    data[0,i,j,:,:], data[1,i,j,:,:], data[2,i,j,:,:])
                    color_space = matlib.repmat(sRGBColor, *data_objects.shape)
                    converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
                        data_objects, color_space)
                    converted2 = np.transpose(np.vectorize(lambda x: (x.rgb_r, x.rgb_g, x.rgb_b))(
                        converted_objects), (1,2,0))
                    converted2 = np.swapaxes(converted2,0,2)
                    converted2 = np.swapaxes(converted2,1,2)
                    # Colormath library interprets rgb in [0,1] and we want [0,255] so let's
                    # normalize to [0,255].
                    converted[:,i,j,:,:] = converted2*255
                    
    else:
        raise Exception('The given input space conversions have not been implemented.')
    if n_d == 2:
        converted = np.squeeze(converted)
    return (converted)

#sample_rgb = results_rgb[0]
#pic_folder_Xrite = pic_folder


def color_calibration(sample_rgb, sample_lab, pic_folder_Xrite, pic_name_Xrite,
                      crop_box_Xrite, offset_array_Xrite):

    
    # Reference data for Xrite color chart in Lab (from
    # http://www.babelcolor.com/colorchecker-2.htm, retrieved in March 2019, for
    # colorchecker passports manufactured after November 2014)
    reference_CC_lab =np.array([[37.54,14.37,14.92],[62.73,35.83,56.5],[28.37,15.42,-49.8],
                                [95.19,-1.03,2.93],[64.66,19.27,17.5],[39.43,10.75,-45.17],
                                [54.38,-39.72,32.27],[81.29,-0.57,0.44],[49.32,-3.82,-22.54],
                                [50.57,48.64,16.67],[42.43,51.05,28.62],[66.89,-0.75,-0.06],
                                [43.46,-12.74,22.72],[30.1,22.54,-20.87],[81.8,2.67,80.41],
                                [50.76,-0.13,0.14],[54.94,9.61,-24.79],[71.77,-24.13,58.19],
                                [50.63,51.28,-14.12],[35.63,-0.46,-0.48],[70.48,-32.26,-0.37],
                                [71.51,18.24,67.37],[49.57,-29.71,-28.32],[20.64,0.07,-0.46]])
    # Reference data is in different order (from upper left to lower left, upper
    # 2nd left to lower 2nd left...). This is the correct order:
    order = list(range(0,21,4)) + list(range(1,22,4)) + list(range(2,23,4)) + list(range(3,24,4))
    reference_CC_lab = reference_CC_lab[order]
    
    # For debugging purposes, let's convert to RGB. --> Was ok!
    #reference_CC_rgb = convert_color_space(reference_CC_lab, 'Lab', 'RGB')
    
    # Let's extract the rgb colors from our Xrite color passport picture.
    CC_rgb = rgb_extractor_Xrite_CC(pic_folder_Xrite, pic_name_Xrite,
                                    crop_box_Xrite, offset_array_Xrite)
    # Convert from RGB to Lab color space.
    CC_lab = convert_color_space(CC_rgb, 'RGB', 'Lab')
    
    
    ###########################
    # Color calibration starts.
    
    # Number of color patches in the color chart.
    N_patches = CC_lab.shape[0]
    
    # Let's create the weight matrix for color calibration using 3D thin plate
    # spline.

    # Data points of our color chart in the original space.
    P = np.concatenate((np.ones((N_patches,1)), CC_lab), axis=1)
    # Data points of our color chart in the transformed space.
    V = reference_CC_lab
    # Shape distortion matrix, K
    K = np.zeros((N_patches,N_patches))
    for i in range(N_patches):
        for j in range(N_patches):
            if i != j:
                r_ij = np.sqrt((P[j,0+1]-P[i,0+1])**2 +
                               (P[j,1+1]-P[i,1+1])**2 +
                               (P[j,2+1]-P[i,2+1])**2)
                U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                K[i,j] = U_ij
    # Linear and non-linear weights WA:
    numerator = np.concatenate((V, np.zeros((4,3))), axis=0)
    denominator = np.concatenate((K,P), axis=1)
    denominator = np.concatenate((denominator,
                                  np.concatenate((np.transpose(P),
                                                  np.zeros((4,4))),axis=1)), axis=0)
    WA = np.matmul(np.linalg.pinv(denominator), numerator)

    # Checking if went ok. We should get the same result than in V (exept for
    # the 4 bottom rows)
    CC_lab_double_transformation = np.matmul(denominator,WA)
    print('Color chart patches in reference Lab:', reference_CC_lab,
          'Color chart patches transformed to color calibrated space and back - this should be the same than above apart from the last 4 rows',
          CC_lab_double_transformation, 'subtracted: ', reference_CC_lab-CC_lab_double_transformation[0:-4,:])
    # --> Went ok!

    n_d = sample_lab.ndim
    if n_d == 3:
        # Let's perform color calibration for the sample points!
        N_samples = sample_lab.shape[0]
        N_times = sample_lab.shape[1]
        sample_lab_cal = np.zeros((N_samples,N_times+4,3))
        # We are recalculating P and K for each sample, but using the WA calculated above.
        for s in range(N_samples):
            # Data points of color chart in the original space.
            P_new = np.concatenate((np.ones((N_times,1)), sample_lab[s,:,:]), axis=1)  ######
            K_new = np.zeros((N_times,N_patches))
            # For each time point (i.e., picture):
            for i in range(N_times):
                # For each color patch in Xrite color chart:
                for j in range(N_patches):
                    #if i != j:
                    r_ij = np.sqrt((P_new[i,0+1]-P[j,0+1])**2 + (P_new[i,1+1]-P[j,1+1])**2 + (P_new[i,2+1]-P[j,2+1])**2)
                    U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                    K_new[i,j] = U_ij
            #sample_lab_cal[s,:,:] = np.matmul(np.concatenate((K_new,P_new),axis=1), WA)
            dennom = np.concatenate((K_new,P_new),axis=1)
            denden = np.concatenate((np.transpose(P), np.zeros((4,4))), axis=1)
            sample_lab_cal[s,:,:] = np.matmul(np.concatenate((dennom, denden), axis=0), WA)
        # Remove zeros, i.e., the last four rows from the third dimension.
        sample_lab_cal = sample_lab_cal[:,0:-4,:]
        ################################
        # Color calibration is done now.
    elif n_d == 5:
        sample_lab_cal = np.zeros(sample_lab.shape)
        print("Main color conversion loop:")
        for x in range(sample_lab.shape[2]):
            print(f"{x}/{sample_lab.shape[2]}")
            for y in range(sample_lab.shape[3]):
                # Let's perform color calibration for the sample points!
                sample_lab2 = sample_lab[:,:,x,y,:]
                N_samples = sample_lab2.shape[1]
                N_times = sample_lab2.shape[2]
                sample_lab_cal2 = np.zeros((3,N_samples,N_times+4))
                # We are recalculating P and K for each sample, but using the WA calculated above.
                for s in range(N_samples):
                    # Data points of color chart in the original space.
                    P_new = np.concatenate((np.ones((N_times,1)), np.swapaxes(sample_lab2[:,s,:],0,1)), axis=1)  ######
                    K_new = np.zeros((N_times,N_patches))
                    # For each time point (i.e., picture):
                    for i in range(N_times):
                        # For each color patch in Xrite color chart:
                        for j in range(N_patches):
                            #if i != j:
                            r_ij = np.sqrt((P_new[i,0+1]-P[j,0+1])**2 + (P_new[i,1+1]-P[j,1+1])**2 + (P_new[i,2+1]-P[j,2+1])**2)
                            U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                            K_new[i,j] = U_ij
                    #sample_lab_cal[s,:,:] = np.matmul(np.concatenate((K_new,P_new),axis=1), WA)
                    dennom = np.concatenate((K_new,P_new),axis=1)
                    denden = np.concatenate((np.transpose(P), np.zeros((4,4))), axis=1)
                    sample_lab_cal2[:,s,:] = np.swapaxes(np.matmul(np.concatenate((dennom, denden), axis=0), WA),0,1)
                # Remove zeros, i.e., the last four rows from the third dimension.
                sample_lab_cal2 = sample_lab_cal2[:,:,0:-4]
                sample_lab_cal[:,:,x,y,:] = sample_lab_cal2
        ################################
        # Color calibration is done now.
    else:
        raise Exception('Faulty number of dimensions in the input!')
    # Let's transform back to rgb.
    sample_rgb_cal = convert_color_space(sample_lab_cal, 'Lab', 'RGB')
    
    
    
    # Let's return both lab and rgb calibrated values.
    return (sample_rgb_cal, sample_lab_cal)

def color_conversion_results(results_rgb):
    # Let's convert these to Lab.
    results_lab = [0,0,0,0, results_rgb[4], 0, 0, results_rgb[7]]
    for i in range(0,4):
        results_lab[i] = convert_color_space(results_rgb[i], 'RGB', 'Lab')
    # Let's plot the data.
    #[results_lab[5], ax_CC, results_lab[6], ax_samples] = plot_results(
    #        results_lab, 'Lab')
    return results_lab
    

def color_calibration_results(results_rgb, results_lab, color_cal_inputs):
    pic_folder = color_cal_inputs[0]
    pic_name_Xrite = color_cal_inputs[1]
    crop_box_Xrite = color_cal_inputs[2]
    offset_array_Xrite = color_cal_inputs[3]
    results_rgb_cal = [0,0,0,0, results_rgb[4], 0, 0, results_rgb[7]]
    results_lab_cal = [0,0,0,0, results_rgb[4], 0, 0, results_rgb[7]]
    for i in range(0,4):
        (results_rgb_cal[i], results_lab_cal[i]) = color_calibration(results_rgb[i], results_lab[i],
                                        pic_folder, pic_name_Xrite,
                                        crop_box_Xrite, offset_array_Xrite)


    # Let's plot the data.
    [results_rgb_cal[5], ax_CC, results_rgb_cal[6], ax_samples] = plot_results(
            results_rgb_cal, 'RGB')
    [results_lab_cal[5], ax_CC, results_lab_cal[6], ax_samples] = plot_results(
            results_lab_cal, 'Lab')
    
    return [results_rgb_cal, results_lab_cal]

def plot_results(results, datatype):
    
    # Let's plot the data.
    t_sort = results[4]
    CC_r_sort = results[3][:,:,0]
    CC_g_sort = results[3][:,:,1]
    CC_b_sort = results[3][:,:,2]
    r_sort = results[0][0,:,:,:,:]
    g_sort = results[0][1,:,:,:,:]
    b_sort = results[0][2,:,:,:,:]
    r_lo_sort = results[1][:,:,0]
    g_lo_sort = results[1][:,:,1]
    b_lo_sort = results[1][:,:,2]
    r_hi_sort = results[2][:,:,0]
    g_hi_sort = results[2][:,:,1]
    b_hi_sort = results[2][:,:,2]
    
    [fig_CC, ax_CC, fig_samples, ax_samples] = plot_aging_data(
                    4,6,t_sort, CC_r_sort, CC_g_sort, CC_b_sort,
                    7,4,r_sort[:,2,2,:], g_sort[:,2,2,:], b_sort[:,2,2,:],
                    r_hi_sort, g_hi_sort, b_hi_sort,
                    r_lo_sort, g_lo_sort, b_lo_sort, datatype)
    
    
    return [fig_CC, ax_CC, fig_samples, ax_samples]



###############################################################################
# Part 2/2: Functions related plotting color calibrated data and saving it as
# pictures.
###############################################################################



def separate_filename_and_folderpath(pic_path):
    # The name of the videos is the name of the aging test.
    slashes = substring_indexes('/', pic_path)
    folder = pic_path[0:slashes[-1]]
    filename = pic_path[(slashes[-1]+1)::]
    return [folder, filename]

def create_pic_name_and_path(pic_path, save_to_folder, name_append, pic_format):
    [folder, filename] = separate_filename_and_folderpath(pic_path)
    new_pic_path = save_to_folder + '/' + filename[0:(-4)] + name_append + '.' + pic_format
    return new_pic_path

def plot_colors(pic_path, crop_box, offset_array, color_array, save_to_folder,
                savefig, name_append, print_out):

   #%%
    # fetch picture for adjusting the cropping box
    # should try find the optimum cropping box options
    testfile = pic_path
     
    image = Image.open(testfile, 'r')
    im = Image.fromarray(np.array(image, dtype=np.uint8), 'RGB')
    ## Create figure and axes
    #fig,ax = plt.subplots(1,figsize=(1248/100,1024/100))
    ## Display the image
    #ax.imshow(im)
    # Create a Rectangle patch
    lw=1 # Line width
    ec='r' # edge color
    fc='none' # face color

    box= crop_box
    rect2 = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                              linewidth=lw,edgecolor=ec,facecolor=fc)
    # Add the patch to the Axes
    #ax.add_patch(rect2)
    #plt.show()
    
    # Color Card Cropping
    [w,h,image_ROI]=get_image(testfile,crop_box)
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    # Row, Columns Settings and Offset pixels for each color square (TO BE CHANGED)
    row_num_CC=7
    col_num_CC=4
    offset_array_CC = offset_array#[[25,25],[25,25]]#[[x_left,x_right],[y_upper,y_lower]]
    ########%%%%%%%%%%%%%%%%%%%%%%%%###############
    [fig_CC, ax_CC, reconstr_CC, image_CC, fig_patches,
     ax_patches]=color_patches_and_image_slicing(
            image_ROI, col_num_CC, row_num_CC, offset_array_CC, color_array)
    
    ax_patches.imshow(Image.fromarray(reconstr_CC, 'RGB'))
    #print('print_out', print_out)
    if print_out == 1:
        plt.show()
    
    if savefig == 1:
        if not os.path.exists(save_to_folder):
            os.makedirs(save_to_folder)
        name1 = create_pic_name_and_path(pic_path, save_to_folder, name_append + '1', 'jpg')
        name2 = create_pic_name_and_path(pic_path, save_to_folder, name_append + '1', 'png')
        #name2 = create_pic_name_and_path(pic_path, save_to_folder, name_append + '2', 'jpg')
        fig_CC.savefig(name1)
        fig_CC.savefig(name2)
        #fig_patches.savefig(name1)
    plt.close(fig_CC)
    return None#(Xrite_rgb)


def color_patches_and_image_slicing(image_array, col_num, row_num, offset_array, color_array):
    """slice the ROIs from an image of an array of samples/colorcard"""
    row_h = int(image_array.shape[0]/row_num)
    col_w = int(image_array.shape[1]/col_num)
    
    fig,ax = plt.subplots(1)#,figsize=(5,5))
    fig_patches, ax_patches = plt.subplots(1)#,figsize=(5,5))
    ax.imshow(Image.fromarray(np.array(image_array, dtype=np.uint8), 'RGB'))

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
            fc='none'
            img = np.transpose(color_array[:,y*col_num + x,:,:]/255, (2,1,0))
            rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                      linewidth=lw,edgecolor=ec,facecolor=fc)
            rect2 = patches.Rectangle(((x2-x1)*x,(y2-y1)*y),x2-x1,y2-y1,
                                      linewidth=lw,edgecolor=ec,facecolor=fc)
            ax.add_patch(rect)
            img = ax.imshow(img, extent=(x1,x2,y1,y2), origin='lower')
            ax_patches.add_patch(rect2)

        imagecol.append(np.concatenate(imagerow, axis=1))
    image_reconstr = np.array(np.concatenate(imagecol, axis=0), dtype=np.uint8)
    ax.imshow(Image.fromarray(np.array(image_array, dtype=np.uint8), 'RGB'), alpha=0)
    return [fig, ax, image_reconstr, images, fig_patches, ax_patches]




#pic_folder='.'
#pic_name='20190328112533_40_np.jpg'
#crop_box=(350+60,250+245,900-80,850-80)
#offset_array=[[20,20],[20,20]]
#color_array=results_rgb[0][:,-1,:]
#testfile = pic_folder+'/'+pic_name
#[w,h,image_ROI]=get_image(testfile,crop_box)

#image_array=image_ROI
#col_num=4
#row_num=7 

#test=plot_colors(pic_folder, pic_name, crop_box, 
#              offset_array, color_array)



