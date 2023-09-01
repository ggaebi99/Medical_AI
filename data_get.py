# -*- coding: utf-8 -*-

import os
import nibabel as nib
from scipy import ndimage
import pydicom
import numpy as np
import natsort
import matplotlib.pyplot as plt

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    
    scan = scan.get_fdata()
    a = 1
    
    if scan.shape[0] == 512:
        scan = ndimage.rotate(scan, 90, reshape=False)
        scan = scan.transpose(2,0,1)
    #for i, y in enumerate(scan):
    #    plt.imsave(data_path2+str(i)+".png",y, cmap=plt.cm.bone)
    #input(1)
    #for i in np.transpose(scan,(2,0,1)):
    #    print(i.shape)
    #    plt.imshow(i, cmap=plt.cm.bone)
    #    plt.show()

    return scan


def normalize(volume):
    
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    
    #for i, y in enumerate(volume):
    #    plt.imsave(data_path2+str(i)+".png",y, cmap=plt.cm.bone)
    #input(2)
    
    # print("정규화 완")
    return volume
    
def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    img = img.transpose(1,2,0)
    desired_depth = 64
    desired_width = 192
    desired_height = 192
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    #for i, y in enumerate(img.transpose(2,0,1)):
    #    plt.imsave(data_path2+str(i)+".png",y, cmap=plt.cm.bone)
    #input(3)
    
    return img



def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    # print(1)
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

data = "data/"

data_list = list()
label_list = list()

for i in os.listdir(data):
    print(i)
    abc = 0
    for y in os.listdir(data+i):
        abc = abc + 1
        # os.rename(data+i+"/"+y,data+i+"/"+y+".nii.gz")
        scan = process_scan(data+i+"/"+y)
        print(scan.shape)
        if i == "covid":
            data_list.append(scan)
            label_list.append(2)
        elif i == "normal":
            data_list.append(scan)
            label_list.append(1)
        else:
            data_list.append(scan)
            label_list.append(3)

    a = 1
    for po in np.transpose(scan,(2,0,1)):
        plt.imsave(i +"_sample/"+str(a)+ ".png",po, cmap=plt.cm.bone)
        a = a+1

data_list = np.array(data_list)
label_list = np.array(label_list)

print(data_list)
print(data_list.shape)
input()
print(label_list)
print(label_list.shape)
input()
np.save('data_list.npy', data_list)
np.save('label_list.npy', label_list)

            
            
            
            