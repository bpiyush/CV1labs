from stl10_input import *
from torch.utils.data import Dataset
import imageio
import glob


# Please run this code snippet before using the STL10 dataset for the first time

# for i in ['test', 'train']:
#     data_path = f'stl10_binary/{i}_X.bin'
#     label_path = f'stl10_binary/{i}_y.bin'
#     images = read_all_images(data_path)
#     labels = read_labels(label_path)
#     save_images(images, labels, dir=f'img/{i}/')


class STL10_Dataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.train = self.mode in ["train", "valid"]
        self.data = []
        self.targets = []
        self.load_data()
        if self.train:
            self.select_samples()

    def load_data(self):

        dirs = self.get_class_dirs()
        for i in range(len(dirs)):
            for im_path in glob.glob(f'{self.root}/img/{"train" if self.train else "test"}/{dirs[i]}/*.png'):
                im = imageio.imread(im_path)
                self.data.append(im)
                self.targets.append(i + 1)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def select_samples(self):
        np.random.seed(0)
        num_train_samples = int(0.8 * len(self.data))
        train_indices = np.random.choice(len(self.data), size=num_train_samples, replace=False)
        if self.mode == 'train':
            self.data = self.data[train_indices]
            self.targets = self.targets[train_indices]
        else:
            all_indices = np.arange(len(self.data))
            val_indices = np.setdiff1d(all_indices, train_indices)
            self.data = self.data[val_indices]
            self.targets = self.targets[val_indices]

    def get_class_dirs(self):
        labels = ['airplane', 'bird', 'ship', 'cat', 'dog']
        with open(f'{self.root}/stl10_binary/class_names.txt') as f:
            names = f.read().splitlines()
        dirs = [names.index(i) + 1 for i in labels]
        return dirs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
