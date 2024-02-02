import os

from torchvision.datasets.folder import ImageFolder, default_loader

from utils.utils import write

class DG(ImageFolder):
    def __init__(self, root, dataset, split_, transform, log_file=None):
        self.root = root
        self.dataset = dataset.replace('-FS', '')

        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if 'train' in split_:
            assert (dataset == 'imagenet') and ('train_shot_16' in split_)
            shot = split_.split('_')[2]
            seed = split_.split('_')[4]
            list_path = os.path.join(self.root, self.dataset, 'annotations/train_meta.list.num_shot_' + shot + '.seed_' + seed)
        elif 'test' in split_:
            list_path = os.path.join(self.root, self.dataset, 'annotations/val_meta.list')
        else:
            raise NotImplementedError

        write('list_path : {}'.format(list_path), log_file=log_file)

        self.samples = []
        with open(list_path, 'r') as f:
            for line in f:
                img_name = line.rsplit(' ', 1)[0]
                label = int(line.rsplit(' ', 1)[1])
                if self.dataset == 'imagenet':
                    if 'train' in split_:
                        self.samples.append((os.path.join(root, self.dataset, 'images/train', img_name), label))
                    elif 'test' == split_:
                        self.samples.append((os.path.join(root, self.dataset, 'images/val', img_name), label))
                    else:
                        raise NotImplementedError
                else:
                    self.samples.append((os.path.join(root, self.dataset, img_name), label))

