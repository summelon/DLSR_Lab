import os
import copy
import tqdm
import argparse as arg
import torch
import torchvision

import my_dataset
import pytorch_warmup as warmup

parser = arg.ArgumentParser()
parser.add_argument(
        '--data_dir', type=str,
        help="Path to dataset, e.g. '../food11re'",
        dest="data_dir", default="../food11re")
parser.add_argument(
        '--balance', type=str,
        help="The way how to balance data. \"weighted\" or \"augment\"",
        dest="balance", default="weighted")
args = parser.parse_args()

if args.data_dir is None:
    raise ValueError("You must supply the dataset dir")
if args.balance is None:
    raise ValueError(
            "Supply a kind of balance method: \"weighted\" or \"augment\"")

BATCH_SIZE = 64
PATIENCE = 2
NUM_EPOCHS = 7
train = 'skewed_training'
validation = 'validation'
evaluation = 'evaluation'
num_cls = 11

device = torch.device(
        "cuda:0"
        if torch.cuda.is_available
        else "cpu")

dataset = {
        x: my_dataset.Food11Dataset(
            os.path.join(args.data_dir, x),
            is_train=(True if x == train else False),
            balance=args.balance
            ) for x in [train, validation, evaluation]}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    data_loaders = {
            x: torch.utils.data.DataLoader(
                dataset=dataset[x],
                num_workers=0,
                shuffle=(
                    True
                    if (x == train and args.balance == 'augment') else False),
                batch_size=BATCH_SIZE,
                sampler=(
                    dataset[x].wts_sampler()
                    if args.balance == 'weighted' else None)
                ) for x in [train, validation, evaluation]}
    print(data_loaders[train].num_workers)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
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
                        scheduler[0].step()
                        scheduler[1].dampen()

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
                            optimizer.param_groups[0]['lr'])
                        )

            # epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / len(dataset[phase])

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

    return model


def main():
    '''
    model_conv = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, num_cls)
    '''
    model_conv = torchvision.models.mobilenet_v2(pretrained=True)
    num_ftrs = model_conv.classifier[1].in_features
    model_conv.classifier[1].in_features = torch.nn.Linear(num_ftrs, num_cls)

    model_conv = model_conv.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(
            model_conv.parameters(),
            lr=9e-3, momentum=0.9)

    # Define number of steps by epoch number
    # Number of steps = number of dataloader * number of epochs
    num_epochs = NUM_EPOCHS
    num_steps = (
            len(dataset[train])//BATCH_SIZE * num_epochs)

    # Decay LR by CosineAnnealing
    cos_anl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_conv, T_max=num_steps)

    # Implement warmup
    warmup_scheduler = warmup.ExponentialWarmup(
            optimizer_conv, warmup_period=27)

    model_conv = train_model(
            model_conv, criterion, optimizer_conv,
            [cos_anl_scheduler, warmup_scheduler],
            num_epochs=num_epochs)

    torch.save(
            model_conv.state_dict(),
            './workspace/t_mobile.pb')


if __name__ == '__main__':
    main()
