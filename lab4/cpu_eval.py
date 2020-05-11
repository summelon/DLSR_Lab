import os
import tqdm
import torch
import torchvision
import numpy as np
import argparse as arg
import my_dataset
import time


parser = arg.ArgumentParser()
parser.add_argument(
        '--data_dir', type=str, default="../food11re/food11re",
        help="Path to dataset, e.g. '../food11re'",
        dest="data_dir")
parser.add_argument(
        '--balance', type=str, default="weighted",
        help="Balance method used when training.",
        dest="balance")
args = parser.parse_args()

if args.data_dir is None:
    raise ValueError("You must supply the dataset dir")

BATCH_SIZE = 1
NUM_CLS = 11
train = 'training'
validation = 'validation'
evaluation = 'evaluation'


device = "cpu"

dataset = my_dataset.Food11Dataset(
            os.path.join("./evaluation"),
            is_train=False,
            balance='weighted',
            img_size=224)


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

    correct_top1 = 0

    for i in range(3347):
        inputs, labels = dataset.__getitem__(i)
        inputs = torch.unsqueeze(inputs, 0).to(device)
        labels = torch.Tensor([labels]).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        correct_top1 += torch.sum(preds == labels.data)
        print(f"{correct_top1}", end="\r")

    result_dict = {"top1": correct_top1.cpu().numpy(),
                   "total": 3347}

    return result_dict


def main():
    model_eval = torchvision.models.resnet18(pretrained=False)

    num_ftrs = model_eval.fc.in_features
    model_eval.fc = torch.nn.Linear(num_ftrs, NUM_CLS)

    model_eval.load_state_dict(torch.load('./model.pt'))
    model_eval.to(device)

    start_time = time.time()
    result = eval_model(model_eval)
    end_time = time.time()

    top1_acc = result["top1"] / 3347
    print("[ INFO ]Accuracy: {:.2f}%.".format(np.average(top1_acc)*100))
    print(f"[ INFO ]Average latency is {1000*(end_time-start_time)/3347:.2f}ms")
    print(f"[ INFO ]FPS is {3347/(end_time-start_time):.2f}")


if __name__ == '__main__':
    main()
