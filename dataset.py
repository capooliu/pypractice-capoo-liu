import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def get_img_info(data_dir, label_mapping):
    imgpath = []
    imglabel = []
    for root, dirs, _ in os.walk(data_dir):
        for sub_dir in dirs:
            if sub_dir in label_mapping:
                sub_dir_path = os.path.join(root, sub_dir)
                img_names = os.listdir(sub_dir_path)
                img_names = [f for f in img_names if f.endswith('.jpg')]
                for img_name in img_names:
                    imgpath.append(os.path.join(sub_dir_path, img_name))
                    imglabel.append(label_mapping[sub_dir])
    return imgpath, imglabel

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)

class CustomTestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)

def denormalize_image(tensor):
    # Denormalize image using standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img
