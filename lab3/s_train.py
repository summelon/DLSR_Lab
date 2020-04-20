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
t_ckpt_path = "./workspace/t_ckpt.pb"

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


def kd_loss(s_outputs, t_outputs, ground_truth, alpha, T):
    kl_distance = torch.nn.KLDivLoss(reduction="batchmean")
    loss1 = kl_distance(
            torch.nn.functional.log_softmax(s_outputs/T, dim=1),
            torch.nn.functional.softmax(t_outputs/T, dim=1))
    loss2 = torch.nn.functional.cross_entropy(s_outputs, ground_truth)
    loss_kd = loss1 * (alpha ** T) + loss2 * (1 - alpha)

    return loss_kd


def train_model(model, optimizer, scheduler, num_epochs=25):
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

    best_model_wts = copy.deepcopy(model[1].state_dict())
    best_acc = 0.0
    patience = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # load best model weights

        # Each epoch has a training and validation phase
        for phase in [train, validation]:
            if phase == train:
                model[1].train()  # Set model to training mode
            else:
                model[1].eval()   # Set model to evaluate mode

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
                    t_outputs = model[0](inputs)
                    s_outputs = model[1](inputs)
                    _, preds = torch.max(s_outputs, 1)
                    # loss = criterion(outputs, labels)
                    loss = kd_loss(
                            s_outputs, t_outputs, labels,
                            alpha=0.95, T=4)

                    # backward + optimize only if on training phase
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
                    best_model_wts = copy.deepcopy(model[1].state_dict())
                    patience = 0
                else:
                    patience += 1

        if patience == PATIENCE:
            break

    print('Best validation Acc: {:4f}'.format(best_acc))
    model[1].load_state_dict(best_model_wts)

    return model[1]


def main():
    teacher_model = torchvision.models.resnet50(pretrained=False)
    student_model = torchvision.models.mobilenet_v2(pretrained=True)
    # for param in model_conv.parameters():
    #    param.requires_grad = False

    # Parameters of newly constructed modules
    # have requires_grad=True by default
    t_num_ftrs = student_model.classifier[1].in_features
    student_model.classifier[1] = torch.nn.Linear(t_num_ftrs, num_cls)
    student_model.to(device)

    s_num_ftrs = teacher_model.fc.in_features
    teacher_model.fc = torch.nn.Linear(s_num_ftrs, num_cls)
    teacher_model.to(device)
    # Load teacher pretrained weight
    teacher_model.load_state_dict(torch.load(t_ckpt_path))
    teacher_model.eval()

    # criterion = torch.nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(
            student_model.parameters(),
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
            [teacher_model, student_model], optimizer_conv,
            [cos_anl_scheduler, warmup_scheduler],
            num_epochs=num_epochs)

    torch.save(
            model_conv.state_dict(),
            './workspace/s_ckpt.pb')


if __name__ == '__main__':
    main()
