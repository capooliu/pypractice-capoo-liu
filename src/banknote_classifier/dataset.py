import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PreprocessStrategy(ABC):
    """Base class for all preprocessing strategies."""

    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Return a composed transform pipeline."""
        pass


class StandardPreprocess(PreprocessStrategy):
    """Basic preprocessing for validation and testing."""

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


class HeavyAugmentation(PreprocessStrategy):
    """Augmented preprocessing for training."""

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def get_img_info(data_dir: str, label_mapping: dict[str, int]) -> tuple[list[str], list[int]]:
    imgpath = []
    imglabel = []
    for root, dirs, _ in os.walk(data_dir):
        for sub_dir in dirs:
            if sub_dir in label_mapping:
                sub_dir_path = os.path.join(root, sub_dir)
                img_names = os.listdir(sub_dir_path)
                img_names = [f for f in img_names if f.endswith(".jpg")]
                for img_name in img_names:
                    imgpath.append(os.path.join(sub_dir_path, img_name))
                    imglabel.append(label_mapping[sub_dir])
    return imgpath, imglabel


class CustomDataset(Dataset):
    def __init__(
        self,
        img_paths: list[str],
        labels: list[int],
        transform: transforms.Compose | PreprocessStrategy | None = None,
    ) -> None:
        self.img_paths = img_paths
        self.labels = labels
        if isinstance(transform, PreprocessStrategy):
            self.transform = transform.get_transforms()
        else:
            self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, int]:
        img = Image.open(self.img_paths[index]).convert("RGB")
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.img_paths)


class CustomTestDataset(Dataset):
    def __init__(
        self,
        img_paths: list[str],
        transform: transforms.Compose | PreprocessStrategy | None = None,
    ) -> None:
        self.img_paths = img_paths
        if isinstance(transform, PreprocessStrategy):
            self.transform = transform.get_transforms()
        else:
            self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor | Image.Image:
        img = Image.open(self.img_paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.img_paths)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    # Denormalize image using standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img
