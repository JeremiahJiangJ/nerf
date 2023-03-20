import cv2
import os


def make_background_transparent(dataset_path, save_path):
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    for file in os.listdir(dataset_path):
        print(file)
        image = cv2.imread(os.path.join(dataset_path, file), 1)
        tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp, 254, 255, cv2.THRESH_BINARY)

        b, g , r = cv2.split(image)

        rgba = [b, g, r ,alpha]

        res = cv2.merge(rgba, 4)

        cv2.imwrite(os.path.join(save_path, file), res)

if __name__ == "__main__":
    dataset_name = "mirror"
    dataset_path = f"../datasets/{dataset_name}/refl_masks_hetnet_msd_cleaned/"
    save_path = f"../datasets/{dataset_name}/refl_masks/"
    make_background_transparent(dataset_path, save_path)

