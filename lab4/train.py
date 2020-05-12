import os
import copy
import tqdm
import torch
import torchvision
from argparse import ArgumentParser

import sys
sys.path.append("/home/chihsheng03/DLSR_Lab/lab4")
import my_dataset
import prune_utils
import eval_utils


PATIENCE = 2
train = 'training'
validation = 'validation'
evaluation = 'evaluation'
num_cls = 11


def train_model(
        model, device, criterion, optimizer,
        data_loaders, scheduler, num_epochs=25):
    # Initialize
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 0
    lr_rec = []

    # Start training
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # load best model weights

        # Each epoch has a training and validation phase
        for phase in [train, validation]:
            if phase == train:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_size = 0

            # Iterate over data.
            pbar = tqdm.tqdm(data_loaders[phase])
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == train:
                        loss.backward()
                        optimizer.step()
                        # Update CosineAnnealing scheduler first,
                        # then warm up scheduler
                        scheduler.step()
                        lr_rec.append(optimizer.param_groups[0]['lr'])

                # statistics
                running_size += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(
                        loss='{:.3f}'.format(
                            running_loss/running_size),
                        acc='{:.3f}'.format(
                            running_corrects.double()/running_size),
                        lr="{:.2e}".format(
                            optimizer.param_groups[0]['lr']),
                        mem="{}".format(
                            torch.cuda.max_memory_allocated(device)//1048576)
                        )

            # epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / running_size

            # deep copy the model
            if phase == validation:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1

        if patience == PATIENCE:
            break

    print('Best validation Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return model


def print_func(l_name, l_type, w_or_b, sps, head=False):
    if head:
        print(f"{'Layer':<25} {'Op type':<15} "
              f"{'params type':<10} {'Sparsity':<10}")
    else:
        print(f"{l_name:<25} {l_type.__class__.__name__:<15} "
              f"{w_or_b:<10} {sps:<10.2%}")

    return True


def run(params):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(params)

    model_conv = torchvision.models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #    param.requires_grad = False

    # Parameters of newly constructed modules
    # have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, num_cls)
    model_conv = model_conv.to(device)

    # Produce dataset
    dataset = {
            x: my_dataset.Food11Dataset(
                os.path.join(params['data_dir'], x),
                is_train=(True if x == train else False),
                balance=params['balance'],
                img_size=params['resolution']
                ) for x in [train, validation, evaluation]}

    # Make data loaders
    data_loaders = {
            x: torch.utils.data.DataLoader(
                dataset=dataset[x],
                num_workers=4,
                shuffle=(
                    True
                    if (x == train and params['balance'] == 'augment')
                    else False),
                batch_size=params['batch_size'],
                sampler=(
                    dataset[x].wts_sampler()
                    if params['balance'] == 'weighted' else None)
                ) for x in [train, validation, evaluation]}

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(
            model_conv.parameters(),
            lr=params['lr'], momentum=0.9)

    # Define number of steps by epoch number
    # Number of steps = number of dataloader * number of epochs
    num_steps = (
            len(dataset[train])//params['batch_size']) * params['num_epochs']

    # Decay LR by CosineAnnealing
    cos_anl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_conv, T_max=num_steps)

    model_conv = train_model(
            model_conv, device,
            criterion, optimizer_conv, data_loaders,
            cos_anl_scheduler,
            num_epochs=params['num_epochs'])

    orig_acc = eval_utils.eval_model(
            model_conv, device, data_loaders[evaluation], num_cls)

    # Pruning model
    layer_collection = prune_utils.get_all_layers(model_conv)
    best_prune_wts = copy.deepcopy(model_conv.state_dict())
    for c in range(100):
        _ = prune_utils.prune(layer_collection, params['pruning_type'])
        model_conv = train_model(
                model_conv, device,
                criterion, optimizer_conv, data_loaders,
                cos_anl_scheduler,
                num_epochs=params['num_epochs'])

        accuracy = eval_utils.eval_model(
                model_conv, device, data_loaders[evaluation], num_cls)

        if accuracy < (orig_acc * params['acc_min_ratio']):
            print(f"Pruning stopped in {c} iteration.")
            break
        else:
            best_prune_wts = copy.deepcopy(model_conv.state_dict())

    model_conv.load_state_dict(best_prune_wts)
    _ = prune_utils.rm_params(layer_collection)
    _ = prune_utils.show_sparse_info(model_conv, print_func)
    torch.save(model_conv.state_dict(), params['save_path'])

    return float(accuracy)


def param_loader():
    parser = ArgumentParser()
    parser.add_argument(
            "--data_dir", type=str, default="../food11re/food11re",
            help="Which folder your dataset is.")
    parser.add_argument(
            "--balance", type=str, default="weighted",
            help="The way how to balance data. \"weighted\" or \"augment\"")
    parser.add_argument(
            "--save_path", type=str, default="./workspace/model.pt",
            help="Save path.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=14,
                        help="Number of total epochs.")
    parser.add_argument("--warmup_period", type=int, default=27,
                        help="Number of epochs for warmup start.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate.")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Resolution of input after augmentation.")
    parser.add_argument("--pruning_type", type=str, default="channel",
                        help="Pruning type.")
    parser.add_argument("--acc_min_ratio", type=float, default=0.9,
                        help="Lowest acceptable accuracy ratio when purning")
    args, _ = parser.parse_known_args()
    return vars(args)


if __name__ == '__main__':
    params = param_loader()
    run(params)
