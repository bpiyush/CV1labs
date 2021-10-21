"""Defines the dataset object for CIFAR-10 dataset."""
from os.path import join
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

import sys; sys.path.append("../")
from utils.io import unpickle


class CIFAR(Dataset):
    """Defines the CIFAR dataset class."""

    def __init__(self, root, train=True, transform=None) -> None:
        self.root = root
        self.train = train
        self.transform = transform

        # load images and targets
        self.load_data(root, train)

    def load_data(self, root, train):
        pattern = "data_batch_*" if train else "test_batch"
        data_pkls = glob(join(root, "cifar-10-batches-py", pattern))

        self.data = []
        self.targets = []

        for pkl in data_pkls:
            data_dict = unpickle(pkl)

            self.data.extend(data_dict[b"data"])
            self.targets.extend(data_dict[b"labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # get sample at index = item
        img, target = self.data[item], self.targets[item]

        # transform the input (e.g. augmentations apply here)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":
    # train dataset
    dataset = CIFAR(root="../datasets/CIFAR-10/", train=True)
    print(f"Dataset images: {dataset.data.shape}")
    assert len(dataset) == dataset.data.shape[0]
    assert len(dataset) == len(dataset.targets)
    img, target = dataset[0]
    print(f"Sample: x ({img.shape}), t ({target})")
    assert img.shape == (32, 32, 3)
    assert target in list(range(10))

    # test dataset
    dataset = CIFAR(root="../datasets/CIFAR-10/", train=False)
    print(f"Dataset images: {dataset.data.shape}")
    assert len(dataset) == dataset.data.shape[0]
    assert len(dataset) == len(dataset.targets)
    img, target = dataset[0]
    print(f"Sample: x ({img.shape}), t ({target})")
    assert img.shape == (32, 32, 3)
    assert target in list(range(10))

    # train dataset with transforms
    from input_transforms import InputTransform
    transform_list = [
        {
            "name": "ToTensor",
            "args": {},
        },
        {
            "name": "Normalize",
            "args": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        },
    ]
    transform = InputTransform(transform_list)
    dataset = CIFAR(root="../datasets/CIFAR-10/", train=True, transform=transform)
    img, target = dataset[0]
    assert img.shape == torch.Size([3, 32, 32])

