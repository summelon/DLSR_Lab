import torch
import torchvision
import torch.nn.utils.prune


def get_all_layers(model_module):
    layer_collection = []
    for module in model_module.modules():
        if len(list(module.children())) > 0:
            continue

        layer_collection.append(module)

    return layer_collection


# TODO prune bias associate with weight
# TODO prune batch normalization
# FIXME Associate with sparse_info() and rm_params()
def prune(layer_col, p_method):
    def pruning(p_m, layer, name, amount):
        if p_m == "element":
            return torch.nn.utils.prune.l1_unstructured(
                    layer, name, amount)
        elif p_m == "channel":
            return torch.nn.utils.prune.ln_structured(
                    layer, name, amount, n=2, dim=0)
        else:
            raise ValueError("Please select \"element\" or \"channel\"")

    for layer in layer_col:
        if isinstance(layer, torch.nn.Conv2d):
            pruning(p_method, layer, name='weight', amount=0.05)
        # if isinstance(layer, torch.nn.Linear):
            # pruning(p_method, layer, name='weight', amount=0.1)

    return True


def sparse_info(m):
    def cal_sparse(layer):
        num_sparse = torch.sum(layer.weight == 0)
        num_total = layer.weight.nelement()
        sparsity = 100. * float(num_sparse) / float(num_total)

        return (num_sparse, num_total, sparsity)

    sparse_info = []
    global_sparse = 0
    global_total = 0

    for layer in m.named_modules():
        if len(list(layer[1].children())) > 0:
            continue
        if isinstance(layer[1], torch.nn.Conv2d):
            # or isinstance(layer[1], torch.nn.Linear):
            layer_result = cal_sparse(layer[1])
            global_sparse += layer_result[0]
            global_total += layer_result[1]
            # Layer name, layer type, layer sparsity
            sparse_info.append((
                layer[0], layer[1].__class__.__name__, layer_result[2]))

    sparse_info.append((
        "global", "all", (100. * float(global_sparse) / float(global_total))))

    return sparse_info


def rm_params(layer_col):
    for layer in layer_col:
        if isinstance(layer, torch.nn.Conv2d):
            # or isinstance(layer, torch.nn.Linear):
            torch.nn.utils.prune.remove(layer, 'weight')

    return True


def main():
    model = torchvision.models.resnet18()
    layer_collection = get_all_layers(model)
    for c in range(3):
        _ = prune(layer_collection, "channel")
        # _ = rm_params(layer_collection)
        for l in sparse_info(model):
            print(l)


if __name__ == "__main__":
    main()
