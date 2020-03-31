import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import PIL
from cus_aug import ImgAugTransform
import my_dataset

mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25


def show_dataset(dataset, n=6):
    img = np.vstack((
        np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
        for i in range(6)))
    print(img.shape)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.savefig("./visual.png")


'''
ds = my_dataset.Food11Dataset(
        "../food11re/skewed_training",
        is_train=True)
'''

imgaug = ImgAugTransform()

transforms = torchvision.transforms.Compose([
    imgaug,
    torchvision.transforms.RandomVerticalFlip()
])

dataset = torchvision.datasets.ImageFolder(
        '../food11re/skewed_training',
        transform=transforms)
show_dataset(dataset)
