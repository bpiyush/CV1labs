"""Defines the STL-10 dataset object."""
from os.path import join
import numpy as np
from numpy.lib.arraysetops import isin

from torch.utils.data import Dataset

from stl10_input import read_all_images, read_labels


LABEL_MAP = {
    1 : "airplanes",
    2 : "birds",
    4 : "cats",
    6 : "dogs",
    9 : "ships",
}


class STL(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.train = self.mode in ["train", "valid"]
        self.transform = transform

        self.load_data(root, mode)
        if self.train:
            self.select_samples()
        
        print(f":::: Loaded dataset from {root}: X ({self.data.shape}) y ({len(self.targets)})")

    def load_data(self, root, mode):
        fname_prefix = "train" if mode in ["valid", "train"] else "test"

        images_path = join(root, "stl10_binary", f"{fname_prefix}_X.bin")
        self.data = read_all_images(images_path)

        labels_path = join(root, "stl10_binary", f"{fname_prefix}_y.bin")
        self.targets = read_labels(labels_path)

        # only consider samples with relevant labels
        relevant_labels = np.array(list(LABEL_MAP.keys()))
        indices = np.where(np.in1d(self.targets, relevant_labels))[0]
        self.data = self.data[indices]
        self.targets = self.targets[indices]


    def select_samples(self, train_fraction=0.8, seed=0):
        np.random.seed(seed)

        num_train_samples = int(train_fraction * len(self.data))
        train_indices = np.random.choice(len(self.data), size=num_train_samples, replace=False)
        all_indices = np.arange(len(self.data))
        valid_indices = np.setdiff1d(all_indices, train_indices)

        if self.mode == "train":
            self.data = self.data[train_indices]
            self.targets = self.targets[train_indices]

        if self.mode == "valid":
            self.data = self.data[valid_indices]
            self.targets = self.targets[valid_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


if __name__ == "__main__":
    root = "../datasets/STL-10/"

    # check train dataset
    dataset = STL(root=root, mode="train")
    x, l = dataset[0]
    assert isinstance(x, np.ndarray) and isinstance(l, np.uint8)

    # check val dataset
    dataset = STL(root=root, mode="valid")

    # check test dataset
    dataset = STL(root=root, mode="test")
    x, l = dataset[0]
    assert isinstance(x, np.ndarray) and isinstance(l, np.uint8)

    