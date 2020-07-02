#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:29:31 2019

@author: armi
"""

'''import cv2
import os

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
'''
import os
import numpy as np

#Find substrings from strings.
def substring_indexes(substring, string):
    last_found = -1  # Begin at -1 so the next position to search from is 0
    all_found = []
    while True:
        # Find next index of substring, by starting after its last found position
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:  
            break  # All occurrences have been found
        all_found.append(last_found)
    return all_found



def create_video_name(folder):
    # The name of the videos is the name of the aging test.
    slashes = substring_indexes('/', folder)
    video_name = 'video'
    if len(slashes) > 1:
        video_name = folder[(slashes[0]+1):slashes[1]]
    elif len(slashes) == 1:
        video_name = folder[(slashes[0]+1)::]
    return video_name
    

def save_as_video(folder, crop_box_samples, crop_box_CC, crop, pic_file_type, video_name = None):
    # The name of the videos is either the name of the aging test (default) or
    # the name given by the user.
    if video_name == None:
        video_name = create_video_name(folder)
    
    # Details for cropping the video
    space = 90 # Extra space around the sample holder and color chart to make the video look more nice.
    width = str(crop_box_samples[2]-crop_box_samples[0] + space) # Width of the video image.
    height = str(crop_box_samples[3] - crop_box_CC[1] + space) #  Height of the video image.
    top_left_x = str(round(crop_box_samples[0] - space/2)) # Top left x coordinate of the video image.
    top_left_y = str(round(crop_box_CC[1] - space/2)) # # Top left y coordinate of the video image.
    # 520:720:320:150    
    
    #framerate = str(round(475000/len(os.listdir(folder)))) # This produces a good framerate for long aging tests.
    framerate = str(round(len(os.listdir(folder))/8)) # This always produces an 8s video
            
    #command = 'ffmpeg -f image2 -framerate 150 -i \'' + folder + '/%*.jpg\' -f mp4 -q:v 0 -vcodec mpeg4 -r 150 ' + video_name + '.mp4'#-r 150 -b 5000k -vcodec mpeg4 -y movie.mp4'
    print(framerate, folder, pic_file_type, video_name)
    command = 'ffmpeg -f image2 -framerate ' + framerate + ' -i \"' + folder + '/%*.' + pic_file_type + '\" -f mp4 -q:v 0 -vcodec mpeg4 -r ' + framerate + ' ' + video_name + '.mp4'#-r 150 -b 5000k -vcodec mpeg4 -y movie.mp4'
    os.system(command)
    
    if crop == 1:
        #    command2 = 'ffmpeg -i ' + video_name + '.mp4 -filter:v \"crop=' + width + ':' + height + ':' + top_left_x + ':' + top_left_y + '\" -c:a copy ' + video_name + '_cropped.mp4'
        #   os.system(command2)
        command3 = 'ffmpeg -f image2 -framerate ' + framerate + ' -i \"' + folder + '/%*.' + pic_file_type + '\" -filter:v \"crop=' + width + ':' + height + ':' + top_left_x + ':' + top_left_y + '\" -f mp4 -q:v 0 -vcodec mpeg4 -r ' + framerate + ' ' + video_name + '_cropped.mp4'#-r 150 -b 5000k -vcodec mpeg4 -y movie.mp4'
        os.system(command3)
    
#    command4 = 'ffmpeg -f image2 -framerate ' + framerate + ' -i \"' + folder + '/%*.jpg\" -filter:v \"crop=' + width + ':' + height + ':' + top_left_x + ':' + top_left_y + ' drawtext=text=%{n}:fontsize=72:r=60:x=(w-tw)/2: y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000099\" -f mp4 -q:v 0 -vcodec mpeg4 -r ' + framerate + ' ' + video_name + '_cropped2.mp4'#-r 150 -b 5000k -vcodec mpeg4 -y movie.mp4'
#    os.system(command4)
    
    
#ffmpeg -f image2 -pattern_type glob -framerate 12 -i 'foo-*.jpeg' -s WxH foo.avi
#folder = './20190401-R1-SS/JPG'
#save_as_video(folder)


'''
os.system("ffmpeg -f image2 -r 1/5 -i ./images/swissGenevaLake%01d.jpg -vcodec mpeg4 -y ./videos/swissGenevaLake.mp4")

Details:
    
    ffmpeg -r 20 -f image2 -i myImage%04d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 myVideo.mp4
    '''