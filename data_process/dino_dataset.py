# refer to: SeCo https://github.com/ServiceNow/seasonal-contrast

import os
from pathlib import Path

import random
import numpy as np
import rasterio
from PIL import Image
import utils.utils as utils
from torch.utils.data import Dataset
from torchvision import transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']

QUANTILES = {
    'min_q': {
        'B2': 3.0,
        'B3': 2.0,
        'B4': 0.0
    },
    'max_q': {
        'B2': 88.0,
        'B3': 103.0,
        'B4': 129.0
    }
}

class MCBase(Dataset):
    def __init__(self, root, bands=None, transform=None):
        super().__init__()
        self.root = Path(root)
        self.bands = bands if bands is not None else 'RGB_BANDS'
        self.transform = transform
        self.dataset = self.get_img_info(root)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        dirs = os.listdir(data_dir)
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(data_dir, sub_dir))
            img_names = list(filter(lambda x: x.endswith('.tif'), img_names))
            img = np.random.choice(img_names, 1)
            path_img = os.path.join(data_dir, sub_dir, img[0])
            data_info.append((path_img, int(sub_dir)))
        return data_info

    @staticmethod
    def get_samples(self):
        dataset = list(self.root.glob('*/*'))
        dataset = list(filter(lambda x: x.endswith('.tif'), dataset))
        return dataset
    
    def __getitem__(self, index):
        path, label = self.dataset[index]
        img = read_image(path, self.bands, QUANTILES)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def read_image(path, bands, quantiles=None):
    if bands == 'RGB_BANDS':
        img = Image.open(path).convert('RGB')
    else:
        channels = []
        for b in bands:
            ch = rasterio.open(path).read(1)
            # ch = rasterio.open(path / f'{b}.tif').read(1)
            if quantiles is not None:
                ch = normalize(ch, min_q=quantiles['min_q'][b], max_q=quantiles['max_q'][b])
            channels.append(ch)
        img = np.dstack(channels)
        img = Image.fromarray(img)
    return img

class MCTemporal(Dataset):
    def __init__(self, root, bands='RGB_BANDS', transform=None):
        super().__init__()
        self.root = Path(root)
        self.bands = bands if bands is not None else 'RGB_BANDS'
        self.transform = transform
        self.samples = self.get_samples(self.root)

    augment = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        utils.GaussianBlur(),
        transforms.RandomHorizontalFlip(),
    ])

    def get_samples(self,data_dir):
        dirs = os.listdir(data_dir)
        return dirs
        # return [(path, idx) for idx, path in enumerate(self.root.glob('*')) if path.is_dir()]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        root = os.path.join(self.root, self.samples[index])
        label = int(root.split('/')[-1])
        img_names = sorted(list(filter(lambda x: x.endswith('.tif'), os.listdir(root))), reverse=True)
        t0, t1, t2 = [read_image(os.path.join(root,path), self.bands, QUANTILES) for path in np.random.choice(img_names, 3)]

        q = t0
        k0 = self.augment(t1)
        k1 = t2
        k2 = self.augment(t0)

        if self.transform is not None:
            img = self.transform([q, k0, k1, k2])

        return img, label
