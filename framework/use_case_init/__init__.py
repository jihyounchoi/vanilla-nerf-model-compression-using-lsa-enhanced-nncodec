import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset, random_split, default_collate
from torch.utils.data import dataloader, dataset
from framework.applications.datasets import imagenet, blender
from framework.applications.utils import evaluation, train, transforms, evaluation_nerf, train_nerf
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelSetting:

    def __init__(self, model_transform, evaluate, train, dataset, criterion):

        self.model_transform = model_transform
        self.evaluate = evaluate
        self.train = train
        self.dataset = dataset
        self.criterion = criterion
        self.train_cf = False

    def init_training(self, dataset_path, batch_size, num_workers):
        train_set = self.dataset(
            root=dataset_path,
            split='train'
        )
        train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=getattr(train_set, "collate_fn", dataloader.default_collate),
                    sampler=getattr(train_set, "sampler", None),
                )

        return train_loader

    def init_test(self, dataset_path, batch_size, num_workers):
        test_set = self.dataset(
            root=dataset_path,
            split='test'
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(test_set, "collate_fn", dataloader.default_collate),
            sampler=getattr(test_set, "sampler", None),
        )
        return test_set, test_loader

    
    def init_validation(self, dataset_path, batch_size, num_workers):
        val_set = self.dataset(
            root=dataset_path,
            split='val'
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(val_set, "collate_fn", dataloader.default_collate),
            sampler=getattr(val_set, "sampler", None),
        )

        return val_set, val_loader

    def init_test_tef(self, dataset_path, batch_size, num_workers, model_name):
        test_set = self.dataset(
            root=dataset_path,
            split='test'
        )
        self.__model_name = model_name

        test_images, test_labels = zip(*test_set.imgs)
        test_loader = tf.data.Dataset.from_tensor_slices((list(test_images), list(test_labels)))
        test_loader = test_loader.map(lambda image, label: tf.py_function(self.preprocess,
                                                                              inp=[image, label],
                                                                              Tout=[tf.float32, tf.int32]), num_parallel_calls=num_workers).batch(
                                                                                batch_size)

        return test_set, test_loader

    def init_validation_tef(self, dataset_path, batch_size, num_workers, model_name):
        val_set = self.dataset(
            root=dataset_path,
            split='val'
        )
        self.__model_name = model_name

        val_images, val_labels = zip(*val_set.imgs)
        val_loader = tf.data.Dataset.from_tensor_slices((list(val_images), list(val_labels)))
        val_loader = val_loader.map(lambda image, label: tf.py_function(self.preprocess,
                                                                              inp=[image, label],
                                                                              Tout=[tf.float32, tf.int32]), num_parallel_calls=num_workers).batch(
                                                                                batch_size)

        return val_set, val_loader

    
    def preprocess(
                    self,
                    image,
                    label
    ):
        image_size = 224

        if self.__model_name == 'EfficientNetB1':
            image_size = 240
        elif self.__model_name == 'EfficientNetB2':
            image_size = 260
        elif self.__model_name == 'EfficientNetB3':
            image_size = 300
        elif self.__model_name == 'EfficientNetB4':
            image_size = 380
        elif self.__model_name == 'EfficientNetB5':
            image_size = 456
        elif self.__model_name == 'EfficientNetB6':
            image_size = 528
        elif self.__model_name == 'EfficientNetB7':
            image_size = 600

        image, label = self.model_transform(image, label, image_size=image_size)

        if 'DenseNet' in self.__model_name:
            return tf.keras.applications.densenet.preprocess_input(image), label
        elif 'EfficientNet' in self.__model_name:
            return tf.keras.applications.efficientnet.preprocess_input(image), label
        elif self.__model_name == 'InceptionResNetV2':
            return tf.keras.applications.inception_resnet_v2.preprocess_input(image), label
        elif self.__model_name == 'InceptionV3':
            return tf.keras.applications.inception_v3.preprocess_input(image), label
        elif self.__model_name == 'MobileNet':
            return tf.keras.applications.mobilenet.preprocess_input(image), label
        elif self.__model_name == 'MobileNetV2':
            return tf.keras.applications.mobilenet_v2.preprocess_input(image), label
        elif 'NASNet' in self.__model_name:
            return tf.keras.applications.nasnet.preprocess_input(image), label
        elif 'ResNet' in self.__model_name and 'V2' not in self.__model_name:
            return tf.keras.applications.resnet.preprocess_input(image), label
        elif 'ResNet' in self.__model_name and 'V2' in self.__model_name:
            return tf.keras.applications.resnet_v2.preprocess_input(image), label
        elif self.__model_name == 'VGG16':
            return tf.keras.applications.vgg16.preprocess_input(image), label
        elif self.__model_name == 'VGG19':
            return tf.keras.applications.vgg19.preprocess_input(image), label
        elif self.__model_name == 'Xception':
            return tf.keras.applications.xception.preprocess_input(image), label

################################# CUSTOM ADDED ############################
""" Added List
Class DummyModelSetting
Class DummyDataset
Class DummyDataLoader
"""

# Define a DummyDataset class that inherits from Dataset
class DummyDataset(Dataset):
    def __init__(self):
        # Initialize any necessary variables or data for the dummy dataset
        pass

    def __len__(self):
        # Return the total number of samples in the dataset
        return 1000  # Adjust the number of samples as needed

    def __getitem__(self, index):
        # Implement the method to retrieve a sample from the dataset given an index
        # Replace this with your desired data generation logic
        sample = torch.randn(3, 224, 224)  # Dummy data with shape (3, 224, 224)
        label = torch.randint(0, 10, (1,))  # Dummy label as integer
        return sample, label

class DummyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
###########################################################################   

class NeRFModelSetting:

    def __init__(self, train):

        self.train = train
            
    def init_training(self):
        train_set = DummyDataset()
        train_loader = DummyDataLoader()

        return train_set, train_loader


    def init_validation(self):
        
        val_set = DummyDataset()
        val_loader = DummyDataLoader()

        return val_set, val_loader


    def init_test(self):
        
        test_set = DummyDataset()
        test_loader = DummyDataLoader()

        return test_set, test_loader
  
        
        
# supported use cases
use_cases = {
    "NNR_PYT":  ModelSetting(None,
                             evaluation.evaluate_classification_model, # Evaluate
                             train.train_classification_model, # You can find it in framework/applications/utils/train.py
                             imagenet.imagenet_dataloaders, # Dataset
                             torch.nn.CrossEntropyLoss() # Criterion
                             ),

    "NNR_TEF": ModelSetting(transforms.transforms_tef_model_zoo,
                            evaluation.evaluate_classification_model_TEF,
                            None,
                            imagenet.imagenet_dataloaders,
                            torch.nn.CrossEntropyLoss
                            ),

    "NERF_PYT" : NeRFModelSetting(train = train_nerf.train_nerf_model)
}