import copy

import torch
import torchvision
import tqdm

from data_loader import FoodDataset

BATCH_SIZE = 128
PATIENCE = 2
data_dir = '../food11re'
train = 'skewed_training'
val = 'validation'
eva = 'evaluation'

device = torch.device(
        "cuda:0"
        if torch.cuda.is_available
        else "cpu")

food_ds = FoodDataset(
        data_dir,
        train, val, eva,
        batch_size=BATCH_SIZE)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    data_sizes = food_ds.sizes()
    data_loaders = food_ds.loaders()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # load best model weights
        model.load_state_dict(best_model_wts)

        # Each epoch has a training and validation phase
        for phase in [train, val]:
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

                # statistics
                running_size += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(
                        loss='{:.3f}'.format(
                            running_loss/running_size),
                        acc='{:.3f}'.format(
                            running_corrects.double()/running_size)
                        )
            if phase == train:
                scheduler.step()
            # epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print(patience)

            # deep copy the model
            if phase == val:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1

        if patience == PATIENCE:
            break
        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    return model


def main():
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules
    # have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, len(food_ds.cls_names()))

    model_conv = model_conv.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.SGD(
            model_conv.fc.parameters(),
            lr=1e-1, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_conv, step_size=2, gamma=0.5)

    model_conv = train_model(
            model_conv, criterion, optimizer_conv,
            exp_lr_scheduler, num_epochs=25)

    torch.save(
            model_conv.state_dict(),
            './workspace/ckpt')


if __name__ == '__main__':
    main()
