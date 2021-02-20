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

def siw_file_metadata(path):
    # For example:
    # path: Train/live/003/003-1-1-1-1.mov
    # path: Train/spoof/003/003-1-2-1-1.mov
    fldr, path = os.path.split(path)
    # live_spoof = os.path.split(os.path.split(fldr)[0])[1]
    path, extension = os.path.splitext(path)
    client_id, sensor_id, type_id, medium_id, session_id = path.split("_")[0].split("-")
    attack_type = {"1": None, "2": "print", "3": "replay"}[type_id]
    if attack_type is not None:
        attack_type = f"{attack_type}/{medium_id}"
    return client_id, attack_type, sensor_id, type_id, medium_id, session_id

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


class SiwDataset(Dataset):

    def __init__(self,dataset_type,dir_path,transform=None,protocol="Protocol_1"):
        self.main_dir = "/storage/alperen/sodecsapp/datasets/SiW"
        self.protocol = protocol
        self.dataset_type = dataset_type
        self.dir_path = dir_path
        self.transform = transform
        self.real_annotation_path = os.path.join(dir_path,protocol,dataset_type,"for_real.lst")
        self.attack_annotation_path = os.path.join(dir_path,protocol,dataset_type,"for_attack.lst")
        self.annotations = []

        def depth_namer(x,i):
            if self.dataset_type == "train":
                return os.path.join(x.replace("Train","Train_depth"),i)
            else: 
                return None

        #### Real Annotations ####
        with open(self.real_annotation_path, 'r') as f:
            # Read the lst file with stripping \n characters
            annotation_list = list(map(str.strip,f.readlines()))
            for annotation in annotation_list:
                video_name, subject_id = annotation.split(" ")


                frame_dir = os.path.join(self.main_dir,video_name)
                for i in os.listdir(frame_dir):
                    self.annotations.append((os.path.join(frame_dir,i), depth_namer(frame_dir,i) , 1))
                
        ### Spoofs ####
        with open(self.attack_annotation_path, 'r') as f:
            # Read the lst file with stripping \n characters
            annotation_list = list(map(str.strip,f.readlines()))
            for annotation in annotation_list:
                video_name, subject_id, attack_type = annotation.split(" ")

                frame_dir = os.path.join(self.main_dir,video_name)
                for i in os.listdir(frame_dir):
                    self.annotations.append((os.path.join(frame_dir,i), None, 0))

        self.annotations = random.choices(self.annotations, k=50000) 
                
        
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
        image_x = seq.augment_image(image_x)         
        
        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample       

"""
class SiwDataLoader:
    def __init__(self, config):

        self.config = config

        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])

        #self.train_dataset = SiwDataset(dataset_type="train",json_path="data/train.json",transform=self.transform_to_tensor)
        #self.val_dataset = SiwDataset(dataset_type="dev",json_path="data/val.json",transform=self.transform_to_tensor)
        #self.test_dataset = SiwDataset(dataset_type="eval",json_path="data/test.json",transform=self.transform_to_tensor)

        if config.data_mode == "json":
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.batch_size, 
                                           shuffle=True, 
                                           num_workers=self.config.data_loader_workers)

            train_len = len(self.train_dataset)
            self.train_iterations = (train_len + self.config.batch_size - 1) // self.config.batch_size
                                    
            self.val_loader = DataLoader(self.val_dataset,
                                           batch_size=self.config.batch_size, 
                                           shuffle=True, 
                                           num_workers=self.config.data_loader_workers)

            val_len = len(self.val_dataset)
            self.val_iterations = (val_len + self.config.batch_size - 1) // self.config.batch_size

            self.test_loader = DataLoader(self.test_dataset,
                                           batch_size=self.config.batch_size, 
                                           shuffle=True, 
                                           num_workers=self.config.data_loader_workers)

            test_len = len(self.test_dataset)
            self.test_iterations = (test_len + self.config.batch_size - 1) // self.config.batch_size
"""
    
if  __name__ == "__main__":
    dataset = SiwDataset(dataset_type="train",dir_path="/storage/alperen/sodecsapp/datasets/SiW/lists",protocol="Protocol_1")
    print(len(dataset))