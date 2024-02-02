import os

from torchvision.datasets.folder import ImageFolder, default_loader

from utils.utils import write

class FGFS(ImageFolder):
    def __init__(self, root, dataset, split_, transform, log_file=None):
        self.root = root
        self.dataset = dataset.replace('-FS', '')

        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if 'train' in split_:
            shot = split_.split('_')[2]
            seed = split_.split('_')[4]
            list_path = os.path.join(self.root, 'few-shot_split', self.dataset, 'annotations/train_meta.list.num_shot_' + shot + '.seed_' + seed)
        elif 'val' in split_:
            list_path = os.path.join(self.root, 'few-shot_split', self.dataset, 'annotations/val_meta.list')
        elif 'test' in split_:
            list_path = os.path.join(self.root, 'few-shot_split', self.dataset, 'annotations/test_meta.list')
        else:
            raise NotImplementedError

        write('list_path : {}'.format(list_path), log_file=log_file)

        self.samples = []
        with open(list_path, 'r') as f:
            for line in f:
                img_name = line.rsplit(' ', 1)[0]
                label = int(line.rsplit(' ', 1)[1])
                self.samples.append((os.path.join(root, self.dataset, img_name), label))

