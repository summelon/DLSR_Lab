from imgaug import augmenters as iaa
import numpy as np
import PIL


class ImgAugTransform:
    def __init__(self, img_size):
        self.aug = iaa.Sequential([
            iaa.Resize((img_size, img_size)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(
                0.25,
                iaa.OneOf([
                    iaa.Dropout(p=(0, 0.1)),
                    iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = PIL.Image.fromarray(img)

        return img
