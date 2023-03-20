import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device('cpu')
num_workers = 1
batch_size  = 2
image_size  = 224

def load_images_from_folder(path):
    '''Load all images in specified folder

    Parameters:
    folder (string): path of folder to load images from

    Returns:
    list: list of images in specified folder
    '''
    images = []
    for filename in os.listdir(path):
        #print(filename)
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images

class ImageData(Dataset):

    def __init__(self, directory, transform = None):
        self.directory = directory
        self.data = load_images_from_folder(directory)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        return image

if __name__ == '__main__':
    augs = A.Compose([A.Resize(height  = image_size, 
                               width   = image_size),
                      A.Normalize(mean = (0), 
                                  std  = (1)),
                      ToTensorV2()])

    image_dataset = ImageData('D:/HetNet/datasets/mirroripad')

    image_loader = DataLoader(image_dataset,
                              batch_size = batch_size,
                              shuffle = False,
                              num_workers = num_workers)

    for batch_idx, inputs in enumerate(image_loader):
        fig = plt.figure(figsize = (14, 7))
        for i in range(4):
            ax = fig.add_subplot(2, 4, i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].numpy())
        break
