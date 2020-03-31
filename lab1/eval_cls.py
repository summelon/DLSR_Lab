import numpy as np
import tqdm
import torch
import torchvision
import argparse as arg
from data_loader import FoodDataset


parser = arg.ArgumentParser()
parser.add_argument(
        '--data_dir',
        type=str,
        help="Path to dataset, e.g. '../food11re'",
        dest="data_dir")
args = parser.parse_args()

if args.data_dir is None:
    raise ValueError("You must supply the dataset dir")

BATCH_SIZE = 128

train = 'skewed_training'
val = 'validation'
eva = 'evaluation'

device = torch.device(
        "cuda:0"
        if torch.cuda.is_available
        else "cpu")

food_ds = FoodDataset(
        args.data_dir,
        train, val, eva,
        batch_size=BATCH_SIZE)


def topk_acc(k, label, output):
    _, i = output.topk(k)
    bingo = i[i.eq(label.data.view(-1, 1))]
    cor_num = torch.tensor(
            (), dtype=torch.int32).new_zeros(len(food_ds.cls_names()))

    for b in bingo:
        cor_num[b.int()] += 1

    return cor_num


def eval_model(model):

    model.eval()

    total_num = torch.tensor(
            (), dtype=torch.int32).new_zeros(len(food_ds.cls_names()))
    correct_top1 = total_num.clone()
    correct_top3 = total_num.clone()

    pbar = tqdm.tqdm(food_ds.loaders()[train])
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
    # Modify the dir error of dataset
    for r in result_dict:
        b = result_dict[r][2]
        result_dict[r] = np.delete(result_dict[r], 2)
        result_dict[r] = np.append(result_dict[r], b)

    return result_dict


def main():
    model_eval = torchvision.models.resnet18(pretrained=False)

    num_ftrs = model_eval.fc.in_features
    model_eval.fc = torch.nn.Linear(num_ftrs, len(food_ds.cls_names()))

    model_eval.load_state_dict(torch.load('./workspace/ckpt'))
    model_eval.to(device)

    result = eval_model(model_eval)

    top1_acc = result["top1"].astype(float) / result["total"].astype(float)
    top3_acc = result["top3"].astype(float) / result["total"].astype(float)
    print("Test data: {}".format(eva))
    print("Top1 accuracy is: {:.2f}%, top3 accuracy is: {:.2f}%.".format(
        np.average(top1_acc)*100,
        np.average(top3_acc)*100))

    for cls in range(len(food_ds.cls_names())):
        print("Class {:<2} {:>4d}/{:<4d} top1 acc: {:.2f}%, {:>4d}/{:<4d} top3 acc: {:.2f}%.".format(
            cls,
            result["top1"][cls], result["total"][cls], top1_acc[cls]*100,
            result["top3"][cls], result["total"][cls], top3_acc[cls]*100))


if __name__ == '__main__':
    main()
