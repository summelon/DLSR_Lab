import os
import glob
import itertools
import copy
import collections
import argparse as arg

import torch
import torchvision.transforms as trans
from PIL import Image

import cus_aug

parser = arg.ArgumentParser()
parser.add_argument(
        "--data_dir", type=str,
        help="Path to dataset.",
        dest="data_dir")
parser.add_argument(
        "--balance", type=str,
        help="Balance method.",
        dest="balance")
args = parser.parse_args()

code2names = {
    0: "Bread",
    1: "Dairy_product",
    2: "Dessert",
    3: "Egg",
    4: "Fried_food",
    5: "Meat",
    6: "Noodles",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable_fruit"
}


def is_image_file(filename):
    return any(
            filename.endswith(extension)
            for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)

    return img


def input_transform():
    imgaug = cus_aug.ImgAugTransform()
    return trans.Compose([
        imgaug,
        trans.ToTensor()])


class Food11Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, is_train=False,
                 input_transform=input_transform,
                 balance=None):
        super(Food11Dataset, self).__init__()
        image_dir = os.path.expanduser(image_dir)
        path_pattern = image_dir + '/**/*.*'
        files_list = sorted(glob.glob(path_pattern, recursive=True))
        self.datapath = image_dir
        record_dict = {int(x): [] for x in os.listdir(image_dir)}
        for file in files_list:
            if is_image_file(file):
                cls_name = int(os.path.basename(file).split("_")[0])
                record_dict[cls_name].append(file)

        self.image_cls_dict = collections.OrderedDict(
                sorted(record_dict.items()))

        self.num_per_classes = {
                cls: len(self.image_cls_dict[cls])
                for cls in self.image_cls_dict}

        self.input_transform = input_transform
        self.is_train = is_train
        if balance == 'weighted':
            # Select weighted sampler when loading data
            pass
        elif balance == 'augment':
            self.augmentation()
        else:
            raise ValueError("Data balance method is not provided.")

        self.image_filenames = list(
                itertools.chain(*self.image_cls_dict.values()))

    def __getitem__(self, index):
        input_file = self.image_filenames[index]
        input = load_img(input_file)
        if self.input_transform:
            input = self.input_transform()(input)
        label = os.path.basename(self.image_filenames[index])
        label = int(label.split("_")[0])
        return input, label

    def __len__(self):
        return len(self.image_filenames)

    def show_details(self):
        for key in sorted(self.num_per_classes.keys()):
            print("{:<8}|{:<20}|{:<12}".format(
                key,
                code2names[key],
                self.num_per_classes[key]
            ))

    def augmentation(self):
        # Only do data augmentation on training data
        if self.is_train:
            print("!!!!!! Using augmentation.")
            total_img = sum(self.num_per_classes.values())
            total_cls = len(self.num_per_classes)
            avg_target = total_img / total_cls
            for cls in self.num_per_classes:
                variance = self.num_per_classes[cls] - int(avg_target)
                # If number of class more than average
                if variance > 0:
                    for _ in range(variance):
                        self.image_cls_dict[cls].remove(
                                self.image_cls_dict[cls][0])
                # If number of class less than average
                elif variance < 0:
                    orig_list = copy.deepcopy(self.image_cls_dict[cls])
                    for add_num in range(abs(variance)):
                        # When do copy operation on small subset,
                        # it happens that index out of range.
                        # Use % to deal with this.
                        add_num = add_num % self.num_per_classes[cls]
                        self.image_cls_dict[cls].append(orig_list[add_num])
                # If number of class equal to average
                else:
                    pass
        # Skip when validation set and evalutation set
        else:
            pass

    def wts_sampler(self):
        # Return weighted sampler when training dataset only
        if self.is_train:
            print("!!!!!!! Using weighted random sampler.")
            ratio = [
                    1./self.num_per_classes[cls]
                    for cls in self.num_per_classes]
            wts = []
            for cls in self.num_per_classes:
                wts += [ratio[cls]] * self.num_per_classes[cls]

            wts = torch.FloatTensor(wts)

            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights=wts,
                    num_samples=sum(self.num_per_classes.values()),
                    replacement=True)

            return sampler
        else:
            return None


def data_loading(loader, dataset):
    num_per_classes = {}
    for batch_idx, (data, label) in enumerate(loader):
        for l in label:
            if l.item() in num_per_classes:
                num_per_classes[l.item()] += 1
            else:
                num_per_classes[l.item()] = 1

    print("*"*50)
    print("Dataset - ", dataset.datapath)
    print("{:<20}|{:<15}|{:<15}".format(
        "class_name", "bf. loading", "af. loading"))
    for key in sorted(num_per_classes.keys()):
        print("{:<20}|{:<15}|{:<15}".format(
            code2names[key],
            dataset.num_per_classes[key],
            num_per_classes[key]
        ))


def main():
    train_datapath = os.path.join(args.data_dir, "skewed_training")
    valid_datapath = os.path.join(args.data_dir, "validation")
    test_datapath = os.path.join(args.data_dir, "evaluation")

    train_dataset = Food11Dataset(
            train_datapath,
            is_train=True,
            balance=args.balance)
    valid_dataset = Food11Dataset(
            valid_datapath,
            is_train=False,
            balance=args.balance)
    test_dataset = Food11Dataset(
            test_datapath,
            is_train=False,
            balance=args.balance)

    ''' For [Lab 2-1] debugging
    train_dataset.augmentation()
    wts = [ 125, 80, 25, 100, 200, 800, 80, 60, 40, 150, 1000 ]
    train_dataset.augmentation(wts)
    '''

    print("*"*50)
    print("Dataset bf. loading - ", train_datapath)
    print(train_dataset.show_details())

    print("*"*50)
    print("Dataset bf. loading - ", valid_datapath)
    print(valid_dataset.show_details())

    print("*"*50)
    print("Dataset bf. loading - ", test_datapath)
    print(test_dataset.show_details())

    if args.balance == "augment":
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                num_workers=4,
                batch_size=8,
                shuffle=True)
    elif args.balance == "weighted":
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                num_workers=4,
                batch_size=8,
                sampler=train_dataset.wts_sampler())
    valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, num_workers=4, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, num_workers=4, batch_size=8, shuffle=False)

    data_loading(train_loader, train_dataset)
    data_loading(valid_loader, valid_dataset)
    data_loading(test_loader, test_dataset)


if __name__ == '__main__':
    main()
