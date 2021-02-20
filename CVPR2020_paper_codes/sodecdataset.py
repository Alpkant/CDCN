"""
SIW Data loader, as given in Mnist tutorial
"""

import json
import imageio as io
import matplotlib.pyplot as plt
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import os
import numpy as np
import random 
from torch.utils.data import DataLoader, TensorDataset, Dataset
import imgaug.augmenters as iaa
import cv2

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

filenameToPILImage = lambda x: Image.open(x)
# img_size = 224


def get_gray_transforms():
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

def get_valid_transforms(img_size=256, norm_mu=[0.485, 0.456, 0.406], norm_sig=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        filenameToPILImage,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mu, norm_sig)
    ])


def imshow(image,depth):

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    image = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 1)
    #image = plt.imshow(depth_image)
    plt.tight_layout()
    ax.set_title("Depth Image")
    ax.axis('off')
    plt.show()


class SodecDataset(Dataset):

    def __init__(self,dataset_type,dir_path,transform=None,protocol="protocol_1"):
        self.main_dir = "/storage/alperen/sodecsapp/datasets/SodecDataset"
        self.protocol = protocol
        self.dataset_type = dataset_type
        self.dir_path = dir_path
        self.transform = transform
        self.annotation_path = os.path.join(self.main_dir,dir_path,protocol+"_"+dataset_type+".tsv")
        self.annotations = []

        #### Real Annotations ####
        with open(self.annotation_path, 'r') as f:
            # Read the lst file with stripping \n characters
            annotation_list = list(map(str.strip,f.readlines()))
            for annotation in annotation_list:
                img_name, subject_name, label = annotation.split("\t")
                img_depth_name = None
                if int(label) == 1:
                    img_depth_name = img_name.replace("/real/","/real_depth/")
                
                self.annotations.append((img_name, img_depth_name , int(label)))                
        
    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, idx):

        img_path, img_depth_path, label = self.annotations[idx]
        
        image_x = np.zeros((256, 256, 3))
        map_x = np.zeros((32, 32))

        image_x = cv2.resize(cv2.imread(img_path), (256, 256))

        if img_depth_path is not None:
            map_x = cv2.resize(cv2.imread(img_depth_path, 0), (32, 32))
             
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        if self.dataset_type == "train":
            image_x = seq.augment_image(image_x)         
        if self.dataset_type == "test":
            np.where(map_x < 1, map_x, 1)

        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': label}

        if self.transform:
            sample = self.transform(sample)


        return sample       

    
if  __name__ == "__main__":
    dataset = SodecDataset(dataset_type="train",dir_path="dataset_with_margin",protocol="protocol_1")
    print(len(dataset))