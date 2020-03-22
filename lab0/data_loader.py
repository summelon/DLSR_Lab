"""
intro
"""
import os

import torch
from torchvision import datasets, transforms


class FoodDataset:
    '''
    data_dir = '../food_dataset'
    train_ds = 'skewed_train'
    val_ds = 'validation'
    eval_ds = 'evaluation'
    '''
    def __init__(
            self, data_dir,
            train_ds, val_ds, eval_ds,
            batch_size):
        self.data_dir = data_dir
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.eval_ds = eval_ds
        self.batch_size = batch_size

        self.data_transforms = {
                self.train_ds: transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                    ]),
                self.val_ds: transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                    ]),
                self.eval_ds: transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                    ]),
                }

        self.image_datasets = {
                x: datasets.ImageFolder(
                    os.path.join(self.data_dir, x),
                    self.data_transforms[x]
                ) for x in [self.train_ds, self.val_ds, self.eval_ds]}

    def loaders(self):
        """
        intro
        """
        dataloaders = {
                x: torch.utils.data.DataLoader(
                    self.image_datasets[x],
                    batch_size=self.batch_size,
                    shuffle=(True if x == self.train_ds else False),
                    num_workers=4
                ) for x in [self.train_ds, self.val_ds, self.eval_ds]}

        return dataloaders

    def sizes(self):
        """
        intro
        """
        dataset_sizes = {
                x: len(self.image_datasets[x])
                for x in [self.train_ds, self.val_ds, self.eval_ds]}

        return dataset_sizes

    def cls_names(self):
        """
        intro
        """
        class_names = self.image_datasets[self.train_ds].classes

        return class_names


if __name__ == "__main__":
    print("data_loader.py is being run directly")
