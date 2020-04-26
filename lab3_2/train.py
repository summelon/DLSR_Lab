import os
import copy
import tqdm
import torch
from argparse import ArgumentParser

import pytorch_warmup as warmup
import sys
sys.path.append("/home/chihsheng03/DLSR_Lab/lab3_2")
#sys.path.append("/home/chihsheng03/.local/lib/python3.6/site-packages")
import my_dataset
import utils


PATIENCE = 2
train = 'skewed_training'
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
                        scheduler[0].step()
                        scheduler[1].dampen()
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

    print('Best validation Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    import matplotlib.pyplot as plt
    x_axis = [x for x in range(len(lr_rec))]
    plt.scatter(x_axis, lr_rec, c='r', marker='.')
    plt.ylabel('learning rate')
    plt.xlabel('steps')
    plt.savefig('lr_record.png')

    return model


def eval_model(model, device, eval_data_loader):
    model.to(device)
    model.eval()

    running_size = 0
    running_corrects = 0

    pbar = tqdm.tqdm(eval_data_loader)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_size += inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        pbar.set_postfix(
                acc='{:.3f}'.format(
                    running_corrects.double()/running_size),
                )

    # epoch_loss = running_loss / data_sizes[phase]
    acc = running_corrects.double() / running_size
    print(running_size)

    return acc


def run(params):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(params)

    model_conv = utils.scalable_resnet(
            layers=params['layers'], width_per_group=params['width'])
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
                num_workers=16,
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

    # Implement warmup
    warmup_scheduler = warmup.ExponentialWarmup(
            optimizer_conv, warmup_period=params['warmup_period'])

    model_conv = train_model(
            model_conv, device,
            criterion, optimizer_conv, data_loaders,
            [cos_anl_scheduler, warmup_scheduler],
            num_epochs=params['num_epochs'])

    accuracy = eval_model(model_conv, device, data_loaders[evaluation])

    # torch.save(model_conv.state_dict(), params['save_path'])

    return float(accuracy)


def param_loader():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../food11re",
                        help="Which folder your dataset is.")
    parser.add_argument(
            "--balance", type=str, default="weighted",
            help="The way how to balance data. \"weighted\" or \"augment\"")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=14,
                        help="Number of total epochs.")
    parser.add_argument("--warmup_period", type=int, default=34,
                        help="Number of epochs for warmup start.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate.")
    parser.add_argument("--layers", nargs='+', type=int, default=[3, 4, 6, 3],
                        help="Model depth.")
    parser.add_argument("--width", type=int, default=64,
                        help="Model width.")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Resolution of input after augmentation.")
    args, _ = parser.parse_known_args()
    return vars(args)


if __name__ == '__main__':
    params = param_loader()
    run(params)
