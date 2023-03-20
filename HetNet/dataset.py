# coding=utf-8

import os
import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std = std
    
    def __call__(self, image, mask=None, edge=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        else:
            mask /= 255
        if edge is None:
            return image, mask
        else:
            edge /= 255
            return image, mask, edge


class RandomCrop(object):
    def __call__(self, image, mask, edge):
        H, W, _ = image.shape
        randw = np.random.randint(W/8)
        randh = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask, edge):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1], edge[:, ::-1]
        else:
            return image, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        else:
            mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        else:
            mask = torch.from_numpy(mask)
            return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        ## INSERT COMPUTING IMAGE MEAN AND STD 
        if self.dataset == 'MSD':
            # MSD
            self.mean = np.array([[[136.67972, 128.13225, 119.58932]]])
            self.std = np.array([[[64.26382, 67.363945, 68.53635]]])
        elif self.dataset == 'PMD':
            # PMD
            self.mean = np.array([[[129.81274, 115.53708, 100.38846]]])
            self.std = np.array([[[65.67121, 65.61214, 67.4352]]])
        else:
            training_dataset_path = self.datapath + '/test'
            training_transforms = transforms.Compose([transforms.Resize((352, 352)), transforms.ToTensor()])
            dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_transforms)
            idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx['image']]
            train_dataset = Subset(dataset, idx)
            train_loader = DataLoader(dataset = train_dataset, batch_size = 3063, shuffle = False)
            mean = 0.
            std = 0.
            total_images_count = 0

            for images, _ in train_loader:
                image_count_in_a_batch = images.size(0)
                print(f"Image count in this batch = {image_count_in_a_batch}")
                #Reshape images from tensor of 4 dim to tensor of 3 dim before calculating mean and std
                images = images.view(image_count_in_a_batch, images.size(1), -1)
                mean += images.mean(2).sum(0)
                std += images.std(2).sum(0)
                total_images_count += image_count_in_a_batch
                print(f'Current Mean = {mean}')
                print(f'Current std = {std}')
                print(f'Current total image count = {total_images_count}')
            mean /= total_images_count
            std /= total_images_count

            mean *= 256
            std *= 256
            self.mean = np.array([[mean.tolist()]])
            self.std = np.array([[std.tolist()]])
            print(self.mean)
            print(self.std)
        
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip()[:-4]) #-4 to exclude the .jpg nabei dk how they write the train.txt file

    def __getitem__(self, idx):
        name = self.samples[idx]
        #print(self.cfg.datapath+'/'+self.cfg.mode+'/mask/'+name+'.png')
        image = cv2.imread(self.cfg.datapath+'/'+self.cfg.mode+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        #image = cv2.imread(self.cfg.datapath+'/'+self.cfg.mode+'/image/'+name)[:,:,::-1].astype(np.float32)
        mask = cv2.imread(self.cfg.datapath+'/'+self.cfg.mode+'/mask/'+name+'.png', 0).astype(np.float32)
        shape = mask.shape
        #shape = image.shape

        if self.cfg.mode == 'train':
            edge = cv2.imread(self.cfg.datapath +'/'+self.cfg.mode+ '/edge/' + name + '.png', 0).astype(np.float32)
            image, mask, edge = self.normalize(image, mask, edge)
            image, mask, edge = self.randomcrop(image, mask, edge)
            image, mask, edge = self.randomflip(image, mask, edge)

            return image, mask, edge
        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name
        #else:
        #    image= self.normalize(image)
        #    image = self.resize(image)
        #    image= self.totensor(image)
        #    return image, shape, name

    def collate(self, batch):
        if self.cfg.dataset == 'MSD':
            size = [224, 256, 288, 320, 352][np.random.randint(0, 5)] # MSD
        elif self.cfg.dataset == 'PMD':
            size = [288, 320, 352, 384, 416][np.random.randint(0, 5)] # PMD
        else:
            return None    
            
        image, mask, edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i] = cv2.resize(edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, mask, edge

    def __len__(self):
        return len(self.samples)
