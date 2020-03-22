import torch
from torchvision import models
from thop import profile


def main():
    resnet50 = models.resnet18()
    mobilenet_v2 = models.mobilenet_v2()

    input_data = torch.randn(1, 3, 224, 224)
    macs_res, params_res = profile(resnet50, inputs=(input_data,))
    macs_mob, params_mob = profile(mobilenet_v2, inputs=(input_data,))

    print("ResNet18    : total MACs: {:>15,}, total params: {:>15,}.".format(
        macs_res, params_res))
    print("MobileNet V2: total MACs: {:>15,}, total params: {:>15,}.".format(
        macs_mob, params_mob))


if __name__ == "__main__":
    main()
