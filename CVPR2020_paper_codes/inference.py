from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from models.CDCNs import Conv2d_cd, CDCN, CDCNpp

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap( x, feature1, feature2, feature3, map_x):
    ## initial images 
    feature_first_frame = x[0,:,:,:].cpu()    ## the middle frame 

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.savefig("result_img.jpg")
    plt.close()


    ## first feature
    feature_first_frame = feature1[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig("result_block1.jpg")
    plt.close()
    
    ## second feature
    feature_first_frame = feature2[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig("result_block2.jpg")
    plt.close()
    
    ## third feature
    feature_first_frame = feature3[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig("result_block3.jpg")
    plt.close()
    
    ## third feature
    heatmap2 = torch.pow(map_x[0,:,:],2)    ## the middle frame 

    heatmap2 = heatmap2.data.cpu().numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap2)
    plt.colorbar()
    plt.savefig("result_depthmap.jpg")
    plt.close()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CDCNpp()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model = model.to(device)


    image = cv2.resize(cv2.imread(args.image_path), (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # normalize into [-1, 1]
    image = (image - 127.5)/128
    # convert to tensor
    image = transforms.functional.to_tensor(image)
    image.unsqueeze_(0)
    image = image.to(device, dtype=torch.float)
    # Get outputs
    depth_map, embedding, cv_block1, cv_block2, cv_block3, input_image = model(image)
    # Save feature maps if you want
    FeatureMap2Heatmap(input_image, cv_block1, cv_block2, cv_block3, depth_map)
    liveness_score = torch.sum(depth_map)
    depth_map = depth_map.cpu().detach().numpy()
    print(f"Liveness Score: {liveness_score:.2f} Threshold = 148.90" )
    prediction = "Real" if liveness_score > 148.90 else "Fake"
    print(f"Prediction: {prediction}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--model-path', type=str, default="models/cdcn_sodec_1.pkl", help='.pkl model checkpoint')
    parser.add_argument('--image-path', type=str, default="test_image.jpeg",help='image path for running an inference')    
    args = parser.parse_args()
    main()