import os
import tqdm
import torch
import torchvision
import numpy as np
import argparse as arg
import my_dataset


parser = arg.ArgumentParser()
parser.add_argument(
        '--data_dir', type=str,
        help="Path to dataset, e.g. '../food11re'",
        dest="data_dir")
parser.add_argument(
        '--balance', type=str,
        help="Balance method used when training.",
        dest="balance")
args = parser.parse_args()

if args.data_dir is None:
    raise ValueError("You must supply the dataset dir")

BATCH_SIZE = 64
NUM_CLS = 11
train = 'training'
validation = 'validation'
evaluation = 'evaluation'

device = torch.device(
        "cuda:0"
        if torch.cuda.is_available
        else "cpu")


dataset = {
        x: my_dataset.Food11Dataset(
            os.path.join(args.data_dir, x),
            is_train=(True if x == train else False),
            balance='weighted',
            img_size=224
            ) for x in [train, validation, evaluation]}


data_loaders = {
        x: torch.utils.data.DataLoader(
            dataset=dataset[x],
            num_workers=4,
            shuffle=False,
            batch_size=BATCH_SIZE,
            sampler=None,
            ) for x in [train, validation, evaluation]}


def topk_acc(k, label, output):
    _, i = output.topk(k)
    bingo = i[i.eq(label.data.view(-1, 1))]
    cor_num = torch.tensor(
            (), dtype=torch.int32).new_zeros(NUM_CLS)

    for b in bingo:
        cor_num[b.int()] += 1

    return cor_num


def eval_model(model):

    model.eval()

    total_num = torch.tensor(
            (), dtype=torch.int32).new_zeros(NUM_CLS)
    correct_top1 = total_num.clone()
    correct_top3 = total_num.clone()

    pbar = tqdm.tqdm(data_loaders[evaluation])
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        correct_top1 += topk_acc(1, labels, outputs)
        correct_top3 += topk_acc(3, labels, outputs)

        for l in labels:
            total_num[l] += 1

    result_dict = {"top1": correct_top1.numpy(),
                   "top3": correct_top3.numpy(),
                   "total": total_num.numpy()}
    '''
    # Modify the dir error of dataset
    for r in result_dict:
        b = result_dict[r][2]
        result_dict[r] = np.delete(result_dict[r], 2)
        result_dict[r] = np.append(result_dict[r], b)
    '''

    return result_dict


def main():
    model_eval = torchvision.models.resnet18(pretrained=False)

    num_ftrs = model_eval.fc.in_features
    model_eval.fc = torch.nn.Linear(num_ftrs, NUM_CLS)

    if args.balance == 'weighted':
        model_eval.load_state_dict(torch.load('./workspace/test_model.pt'))
    elif args.balance == 'augment':
        model_eval.load_state_dict(torch.load('./workspace/test_model.pt'))
    model_eval.to(device)

    result = eval_model(model_eval)

    top1_acc = result["top1"].astype(float) / result["total"].astype(float)
    top3_acc = result["top3"].astype(float) / result["total"].astype(float)
    print("Test data: {}".format(evaluation))
    print("Top1 accuracy is: {:.2f}%, top3 accuracy is: {:.2f}%.".format(
        np.average(top1_acc)*100,
        np.average(top3_acc)*100))

    for cls in range(NUM_CLS):
        print("Class {:<2} {:>4d}/{:<4d} top1 acc: {:.2f}%, {:>4d}/{:<4d} top3 acc: {:.2f}%.".format(
            cls,
            result["top1"][cls], result["total"][cls], top1_acc[cls]*100,
            result["top3"][cls], result["total"][cls], top3_acc[cls]*100))


if __name__ == '__main__':
    main()
