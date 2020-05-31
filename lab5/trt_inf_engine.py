import os
import time
import tqdm
import glob
import numpy as np
import tensorrt as trt
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

import common


def data_loader(p, bs, img_size):
    image_list = glob.glob(os.path.join(p, '*/*'))

    for batch in range(0, len(image_list), bs):
        batch_image_name = image_list[batch:batch+bs]

        if len(batch_image_name) != bs:
            batch_image_name += [
                    batch_image_name[0]] * (bs - len(batch_image_name))

        batch_ground_truth = [int(name.split('/')[-1].split('_')[0])
                              for name in batch_image_name]
        batch_image_data = preprocessing(batch_image_name, crop_size=img_size)

        yield (batch_image_data, batch_ground_truth)


def preprocessing(image_name: list, crop_size: tuple) -> list:
    def center_crop(image):
        # image = cv2.resize(image, (crop_size[0]+32, crop_size[1]+32))
        top_left = [(image.size[0]-crop_size[0])//2,
                    (image.size[1]-crop_size[1])//2]

        return image.crop((top_left[0], top_left[1],
                          top_left[0]+crop_size[0], top_left[1]+crop_size[1]))

    image_batch = np.ndarray(
            shape=(len(image_name), 3, crop_size[0], crop_size[1]),
            dtype=np.float32, order='C')
    for i, name in enumerate(image_name):
        # image_data = cv2.imread(name)[:, :, ::-1]
        image_data = Image.open(name).resize(
                (crop_size[0]+32, crop_size[1]+32))
        image_data = np.array(
                center_crop(image_data), dtype=np.float32, order='C')
        image_data = image_data / 255.
        image_data = (
                image_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # Change data layout from HWC to CHW
        image_data = image_data.transpose((2, 0, 1))
        image_batch[i] = image_data

    return image_batch


# NOTE query the builder to find out what mixed-precision types are natively
#      supported by the hardware.
# NOTE Serialize an engine or not?
# NOTE Native trt not support INT64, while my ONNX is INT64, should be INT32
# Reference: https://github.com/rmccorm4/tensorrt-utils/blob/master/
#               classification/imagenet/onnx_to_tensorrt.py
def build_engine(bs, model):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(common.EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        # builder.max_workspace_size = 3 << 30
        config.max_workspace_size = 3 << 30
        # Implicit batch
        builder.max_batch_size = bs

        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 224, 224),
                          (4, 3, 224, 224), (64, 3, 224, 224),)
        config.add_optimization_profile(profile)
        with open(model, 'rb') as model:
            if not parser.parse(model.read()):
                print("Big error here.")

        # network.get_input(0).shape = [bs, 3, 224, 224]

        # Add write engine

        return builder.build_engine(network, config)


def iter_inf(ds_path, batch_size, model):
    dataloader = data_loader(ds_path, batch_size, (224, 224))
    num_cls = len(glob.glob(os.path.join(ds_path, '*')))
    rslts = {}
    rslts['eval_num'] = len(glob.glob(os.path.join(ds_path, '*/*')))
    correct, counter, inf_time = 0, 0, 0.0
    batch_num = rslts['eval_num'] // batch_size + 1
    remainder = rslts['eval_num'] % batch_size

    print("[ INFO ] Building engine.")
    with build_engine(batch_size, model) as engine, \
            engine.create_execution_context() as context:

        rslts['warm_up_start'] = time.time()
        # For multi profile
        # context.active_optimization_profile = 0
        context.set_binding_shape(0, (batch_size, 3, 224, 224))
        print("[ INFO ] Inference start.")
        pbar = tqdm.tqdm(dataloader)
        for batch in pbar:
            imgs, labels = batch

            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = imgs
            counter += 1

            inf_start = time.time()
            trt_outputs = common.do_inference_v2(
                    context, bindings, inputs, outputs, stream)
            inf_time += time.time() - inf_start

            splited_outputs = ([[output[i:i+num_cls]
                                 for i in range(0, len(output), num_cls)]
                                for output in trt_outputs])
            preds = [np.argsort(splited_output)[:, -1]
                     for splited_output in splited_outputs]

            # FIXME Only count first output here
            if counter != batch_num:
                correct += np.sum(np.equal(preds[0], labels))
            else:
                correct += np.sum(
                        np.equal(preds[0][:remainder], labels[:remainder]))

    rslts['inf_time'] = inf_time
    rslts['end'] = time.time()
    rslts['correct'] = correct
    print("[ INFO ] Inference done.")

    return rslts


def show_performance(rslts: dict):
    warm_up_start, inf_time = rslts['warm_up_start'], rslts['inf_time']
    end, correct = rslts['end'], rslts['correct']
    eval_num = rslts['eval_num']
    print(f"[ INFO ] Total time is {end-warm_up_start:.2f}s")
    print(f"[ INFO ] Warm up time is {end-warm_up_start-inf_time:.2f}s")
    print(f"[ INFO ] Inference time is {inf_time:.2f}s")
    print(f"[ INFO ] FPS is {eval_num/(end-warm_up_start):.2f}")
    print(f"[ INFO ] FPS(exclude warmup) is {eval_num/(inf_time):.2f}")

    print(f"[ INFO ] Accuracy is {correct/eval_num:.2%}")


def draw_plot(bs: list, fps: list, fps_wo_wu: list, acc):
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle('TensorRT Performance under Various Batch Size',
                 fontweight='bold')
    ax.set_title(f"Accuracy is {acc:.2%}")

    ax.plot(bs, fps, '-o', label="FPS w/ warm up")
    ax.plot(bs, fps_wo_wu, '-^', label="FPS w/o warm up")
    ax.set_xlabel('Batch size')
    ax.set_ylabel('FPS')

    plt.legend(loc='best')
    plt.xticks(bs)
    plt.savefig('./result.png')


def run(params):
    Image.MAX_IMAGE_PIXELS = 2300000000
    bs_list = [1, 2, 4, 8, 16, 32, 64]
    fps_list, fps_wo_wu_list = [], []

    for batch_size in bs_list:
        print('-'*66)
        print(f"[ INFO ] Do inference in batch size {batch_size}.")
        results = iter_inf(params['dataset_path'],
                           batch_size, params['model_path'])
        show_performance(results)
        fps = results['eval_num']/(results['end']-results['warm_up_start'])
        fps_wo_wu = results['eval_num']/results['inf_time']
        fps_list.append(fps)
        fps_wo_wu_list.append(fps_wo_wu)

    draw_plot(bs_list, fps_list, fps_wo_wu_list,
              results['correct']/results['eval_num'])


def params_loader():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./evaluation",
                        help="Path to evaluation dataset")
    parser.add_argument("--model_path", type=str,
                        default="./special_model.onnx",
                        help="Path to special ONNX model")
    args, _ = parser.parse_known_args()
    params = {k: v for k, v in vars(args).items() if v is not None}

    return params


if __name__ == "__main__":
    p = params_loader()
    run(p)
