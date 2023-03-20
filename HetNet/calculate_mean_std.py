import os
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

training_dataset_path = './datasets/MSD/train/'
#training_dataset_path = './datasets/PMD/train/'
#training_dataset_path = './datasets/mirroripad/test/'

training_transforms = transforms.Compose([transforms.Resize((352, 352)), transforms.ToTensor()])
#training_transforms = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_transforms)
idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx['image']]
train_dataset = Subset(dataset, idx)
train_loader = DataLoader(dataset = train_dataset, batch_size = 3063, shuffle = False)

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        print(f"Image count in this batch = {image_count_in_a_batch}")
        #Reshape images from tensor of 4 dim to tensor of 3 dim before calculating mean and std
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
        print(f'Current Mean = {mean}')
        print(f'Current std = {std}')
        print(f'Currnet total image count = {total_images_count}')

    mean /= total_images_count
    std /= total_images_count

    mean *= 256
    std *= 256
    return np.array([[mean.tolist()]]), np.array([[std.tolist()]])

mean, std = get_mean_and_std(train_loader)

print(f"Mean = {mean}")
print(f"Std = {std}")
