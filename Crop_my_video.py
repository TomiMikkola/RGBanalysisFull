import os
import numpy as np

def crop_video(video_file, width, height, top_left_x, top_left_y, new_video_name):
    
    # Details for cropping the video
    #space = 10 # Extra space around the crop_box to make the video look more nice.
    #width = str(crop_box_samples[2]-crop_box_samples[0] + space) # Width of the video image.
    #height = str(crop_box_samples[3] - crop_box_CC[1] + space) #  Height of the video image.
    #top_left_x = str(round(crop_box_samples[0] - space/2)) # Top left x coordinate of the video image.
    #top_left_y = str(round(crop_box_CC[1] - space/2)) # # Top left y coordinate of the video image.
    # 520:720:320:150    
    
    framerate = str(round(len(os.listdir(folder))/8)) # This always produces an 8s video
    command = 'ffmpeg -i ' + video_file + ' -filter:v "crop=' + width + ':' + height + ':' + top_left_x + ':' + top_left_y + '" ' + new_video_name 
    os.system(command)
    
    
video_file = ''
width = 200
height = 200
top_left_x = 400
top_left_y = 400
crop_video(video_file, width, height, top_left_x, top_left_y, new_video_name)

