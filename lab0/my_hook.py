from torchvision import models
import torch
import numpy as np
import argparse as arg

model_params = 0
model_macs = 0

parser = arg.ArgumentParser()
parser.add_argument(
        '--model_name',
        type=str,
        help="resnet18 or mobilenet_v2",
        dest="model",
        default="resnet18")
args = parser.parse_args()


def get_all_layers(model_module):
    layer_collection = []
    for module in model_module.modules():
        if len(list(module.children())) > 0:
            continue

        layer_collection.append(module)

    return layer_collection


def print_info(self, i_m, o_m, m, f):
    n = self._get_name()
    i = list(i_m[0].size())
    o = list(o_m.size())
    p = sum(x.numel() for x in self.parameters())
    global model_params, model_macs
    model_params += sum(x.numel() for x in self.parameters())
    model_macs += m
    print("{:<17}{:>17}{:>17}{:>10}{:>15}{:>15}".format(
        n, str(i), str(o), p, m, f))


# Layer hook function--------------------------------------
def conv2d(self, input_mat, output_mat):
    """
    Ignore -(Cout*H*W) because of element wise add
    """
    kernel_ele = np.prod(self.weight.size()[1:])
    macs = output_mat.numel() * (kernel_ele + (1 if self.bias else 0))
    flops = macs * 2

    print_info(self, input_mat, output_mat, macs, flops)


def batch_norm(self, input_mat, output_mat):
    """
    Only calculate when evaluation
    BN: consider (sub, div), (sqrt, add) as two MACs
    """
    if not self.training:
        macs = 2 * output_mat.numel()
    else:
        macs = 0
    flops = macs * 2

    print_info(self, input_mat, output_mat, macs, flops)


def linear(self, input_mat, output_mat):
    """
    Linear: (Cin*k^2-1(+1 if bias))*Cout*H*W*N
             (Cin +(0 if bias else -1))*Cout*N
    """
    batch_size = list(output_mat.size())[0]
    out_shape, in_shape = self.weight.size()
    bias_part = 0 if self.bias is not None else -1
    macs = batch_size * (in_shape + bias_part) * out_shape
    flops = macs * 2

    print_info(self, input_mat, output_mat, macs, flops)


def ada_avg_pool(self, input_mat, output_mat):
    """
    stride      = feature_in // feature_out
    padding     = 0
    kernel_size = feature_in + 2 * padding - stride * (feature_out - 1)
    ---------------------
    Add   = kernel_size^2 - 1
    Div   = 1
    FLOPs = (Add + Div) * Cout*H*W*N
          = kernel_size^2 * Cout*H*W*N
    """
    out_shape = list(output_mat.size())
    f_in = list(input_mat[0].size())[2]
    f_out = out_shape[2]
    stride = f_in // f_out
    k_size = f_in - stride * (f_out - 1)
    macs = 1 * np.prod(out_shape)
    flops = k_size**2 * np.prod(out_shape)

    print_info(self, input_mat, output_mat, macs, flops)


def zero_layer(self, input_mat, output_mat):
    macs = 0
    flops = 0
    print_info(self, input_mat, output_mat, macs, flops)


def unknow_layer(self, input_mat, output_mat):
    print("-------Op {} is not in the collection.".format(self._get_name()))


# ---------------------------------------------------------------
def assign_hook(layer_collection):
    hook_fn_collection = {
            torch.nn.Conv2d: conv2d,
            torch.nn.BatchNorm2d: batch_norm,
            torch.nn.AdaptiveAvgPool2d: ada_avg_pool,
            torch.nn.Linear: linear,
            torch.nn.ReLU: zero_layer,
            torch.nn.ReLU6: zero_layer,
            torch.nn.MaxPool2d: zero_layer,
            torch.nn.Dropout: zero_layer}
    handle_buffer = []
    for layer in layer_collection:
        if type(layer) in hook_fn_collection:
            hook_fn = hook_fn_collection[type(layer)]
        else:
            hook_fn = unknow_layer
        handle = layer.register_forward_hook(hook_fn)
        handle_buffer.append(handle)

    return handle_buffer


def main():
    if args.model not in ["resnet18", "mobilenet_v2"]:
        raise ValueError("Please read the help info.")
    if args.model == "resnet18":
        model = models.resnet18()
    elif args.model == "mobilenet_v2":
        model = models.mobilenet_v2()
    model.eval()
    layers = get_all_layers(model)
    handles = assign_hook(layers)

    print("Test model: {}".format(model.__class__.__name__))
    print("{:^17}{:^17}{:^17}{:>10}{:>15}{:>15}".format(
        "Op_name",
        "Input_shape", "Output_shape",
        "Params", "MACs", "FLOPs"))

    input_data = torch.randn(1, 3, 224, 224)
    _ = model(input_data)

    _ = [h.remove() for h in handles]

    print("Total params is: {}, total MACs is: {}".format(
        model_params, model_macs))


if __name__ == '__main__':
    main()
