
import os

import pandas as pd
from torchvision import datasets, transforms

from framework.applications import settings

VALIDATION_FILES = os.path.join(settings.METADATA_DIR, 'imagenet_validation_files.txt')

transforms_pyt_model_zoo = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, root, *args, validate=False, train=True, use_precomputed_labels=False,
                 labels_path=None, **kwargs):
        """ImageNet root folder is expected to have two directories: train and val."""
        if train and validate == train:
            raise ValueError('Train and validate can not be True at the same time.')
        if use_precomputed_labels and labels_path is None:
            raise ValueError('If use_precomputed_labels=True the labels_path is necessary.')

        if train:
            root = os.path.join(root, 'train')
        elif validate:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
            
        super().__init__(root, transform=transforms_pyt_model_zoo, *args, **kwargs)

        if validate and use_precomputed_labels:
            df = pd.read_csv(labels_path, sep='\t')
            df.input_path = df.input_path.apply(lambda x: os.path.join(root, x))
            mapping = dict(zip(df.input_path, df.pred_class))
            # self.samples = [(mapping[x[0]], x[1]) for x in self.samples]
            self.samples = [(x[0], mapping[x[0]]) for x in self.samples]
            self.targets = [x[1] for x in self.samples]

        if validate:
            with open(VALIDATION_FILES, 'r') as f:
                names = [x.strip() for x in f.readlines()]
            class_names = [x.split('_')[0] for x in names]
            val_names = set(os.path.join(self.root, class_name, x) for class_name, x in zip(class_names, names))
            self.samples = [x for x in self.samples if x[0] in val_names]
            self.targets = [x[1] for x in self.samples]


        if train:
            with open(VALIDATION_FILES, 'r') as f:
                names = [x.strip() for x in f.readlines()]
            class_names = [x.split('_')[0] for x in names]
            val_names = set(os.path.join(self.root, class_name, x) for class_name, x in zip(class_names, names))
            self.samples = [x for x in self.samples if x[0] not in val_names]
            self.targets = [x[1] for x in self.samples]


def imagenet_dataloaders(root, split='test'):

    if split == 'train':
        train_data = ImageNetDataset(root=root,
                                     train=True,
                                     validate=False
                                     )
        return train_data

    elif split == 'val':
        val_data = ImageNetDataset(root=root,
                                   train=False,
                                   validate=True
                                   )
        return val_data

    elif split == 'test':
        test_data = ImageNetDataset(root=root,
                                    train=False,
                                    validate=False
                                    )
        return test_data
