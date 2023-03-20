from random import choices
import numpy as np
import os
import time
import cv2
import argparse
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import skimage.io
#from config import msd_testing_root
#from config import testing_root
from misc import check_mkdir, crf_refine
from Net import Net

#from model.pmd import PMD
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset to generate reflection masks from")
parser.add_argument("--mask_gen_type", type=str, default="msd", help="Use HetNet model trained on MSD/PMD dataset", choices=['msd','pmd'])
parser.add_argument("--clean_mask", action='store_true', help="perfrom further preprocessing on masks generated")
parser.add_argument("--transparent_bg", action='store_true', help="makes the background transparent")

def make_background_transparent(dataset_path, save_path):
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    count = 0
    for file in os.listdir(dataset_path):
        count += 1
        print(f"Making background transparent: {count}/{len(os.listdir(dataset_path))}")
        image = cv2.imread(os.path.join(dataset_path, file), 1)
        tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp, 254, 255, cv2.THRESH_BINARY)

        b, g , r = cv2.split(image)

        rgba = [b, g, r ,alpha]

        res = cv2.merge(rgba, 4)

        cv2.imwrite(os.path.join(save_path, file), res)


def clean_mask(datapath, savepath, lower=254, upper=255):
    if os.path.isdir(datapath) == False:
        print(f"{datapath} is INVALID")
        return

    if os.path.isdir(savepath) == False:
        os.mkdir(savepath)
    count = 0
    for file in os.listdir(datapath):
        count += 1
        print(f"Cleaning Mask: {count}/{len(os.listdir(datapath))}")
        ### THRESHOLD BINARIZE IMAGE ###
        image = cv2.imread(os.path.join(datapath, file), cv2.IMREAD_UNCHANGED)
        th, image_th = cv2.threshold(image, lower, upper, cv2.THRESH_BINARY)

        ### FIND LARGEST CONTOUR AND DISCARD THE REST ###
        contours = cv2.findContours(image_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        max_area = 0
        max_contour_idx = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.moments(contour)['m00']
            if contour_area > max_area:
                max_area = contour_area
                max_contour_idx = i

        largestCC_image = np.zeros(image_th.shape, dtype = np.uint8) ## blank image
        cv2.drawContours(largestCC_image, contours, max_contour_idx, color = 255, thickness = -1) #draw in largset cc
        cv2.imwrite(os.path.join(savepath, file), largestCC_image)

def main():
    args = parser.parse_args()
    dataset_name = args.dataset_name
    dataset_type = args.mask_gen_type

    save_path = "../datasets"
    save_folder_og = f"refl_masks_hetnet_{dataset_type}_og"
    save_folder_cleaned = f"refl_masks_hetnet_{dataset_type}_cleaned"
    save_folder_transbg = "refl_masks"
    save_folder_edge = f"edge_hetnet_{dataset_type}"

    device_ids = [0]
    torch.cuda.set_device(device_ids[0])
    print(f"Using CUDA Device {device_ids[0]}")

    testing_root = f"../datasets/{dataset_name}/"
    model_best = f"./models/{dataset_type.upper()}-model-best/model-best"

    if os.path.isdir(model_best[:-10]):
        print(f"{model_best[:-10]} exists")
    else:
        print(f"{model_best[:-10]} does not exist")

    to_test = {dataset_name : testing_root}

    img_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    hetnet = Net().cuda(device_ids[0])
    hetnet.load_state_dict(torch.load(model_best))

    hetnet.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'images'))]
            start_time = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                check_mkdir(os.path.join(save_path, name, save_folder_og))
                #check_mkdir(os.path.join(save_path, name, save_folder_edge))
                img = Image.open(os.path.join(root, 'images', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                out, out_edge = hetnet(img_var, (h, w))

                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                #edge = (torch.sigmoid(out_edge[0, 0]) * 255).cpu().numpy()

                cv2.imwrite(os.path.join(save_path, name, save_folder_og, img_name[:-4] + ".png"), np.round(pred))
                #cv2.imwrite(os.path.join(save_path, name, save_folder_edge, img_name[:-4] + ".png"), np.round(edge))
            
            end_time = time.time()
            print("Average Time Is : {:.2f}".format((end_time - start_time) / len(img_list)))
    if(args.clean_mask):
        print("Cleaning Masks")
        clean_mask(os.path.join(save_path, dataset_name, save_folder_og), os.path.join(save_path, dataset_name, save_folder_cleaned))
    if(args.transparent_bg):
        print("Making Background Transparent")
        make_background_transparent(os.path.join(save_path, dataset_name, save_folder_cleaned), os.path.join(save_path, dataset_name, save_folder_transbg))
if __name__ == '__main__':
    main()