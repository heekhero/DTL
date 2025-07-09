import os

from torchvision.datasets.folder import ImageFolder, default_loader

class VTAB(ImageFolder):
    def __init__(self, root, dataset, split_, transform):

        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if dataset == 'cifar_100':
            data_map = 'cifar'
        elif dataset == 'pets':
            data_map = 'oxford_iiit_pet'
        elif dataset == 'flowers102':
            data_map = 'oxford_flowers102'
        else:
            data_map = dataset
        self.dataset_root = os.path.join(root, data_map)

        if split_ == 'train_val':
            list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        elif split_ == 'train':
            list_path = os.path.join(self.dataset_root, 'train800.txt')
        elif split_ == 'val':
            list_path = os.path.join(self.dataset_root, 'val200.txt')
        elif split_ == 'test':
            list_path = os.path.join(self.dataset_root, 'test.txt')
        else:
            raise NotImplementedError

        self.samples = []

        with open(list_path, 'r') as f:
            for line in f:
                img_name = line.split(' ')[0]
                label = int(line.split(' ')[1])
                self.samples.append((os.path.join(self.dataset_root, img_name), label))