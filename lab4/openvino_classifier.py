from __future__ import print_function
import glob
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS
from PIL import Image
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument(
            '-h', '--help', action='help', default=SUPPRESS,
            help='Show this help message and exit.')
    args.add_argument(
            "-m", "--model", required=True, type=str,
            help="Required. Path to an .xml file with a trained model.")
    args.add_argument(
            "-i", "--input", required=True, type=str,
            help="Required. "
                 "Path to a folder with images or path to an image files")
    args.add_argument(
            "-l", "--cpu_extension", type=str, default=None,
            help="Optional. Required for CPU custom layers. "
                 "MKLDNN (CPU)-targeted custom layers. "
                 "Absolute path to a shared library with the"
                 " kernels implementations.")
    args.add_argument(
            "-d", "--device", default="CPU", type=str,
            help="Optional. Specify the target device to infer on;"
                 "CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. "
                 "The sample will look for a suitable plugin for "
                 "device specified. Default value is CPU")
    args.add_argument(
            "--labels", default=None, type=str,
            help="Optional. Path to a labels mapping file",)
    args.add_argument(
            "-nt", "--number_top", default=10, type=int,
            help="Optional. Number of top results")
    args, _ = parser.parse_known_args()

    return vars(args)


# NOTE batch size is associated with net.batch_size
# FIXME batch size should be h, w
def data_loader(path, batch_size, img_size):
    image_list = glob.glob(path+'/*/*')

    for batch in range(0, len(image_list), batch_size):
        batch_image_name = image_list[batch:batch+batch_size]
        batch_ground_truth = [int(name.split('/')[-1].split('_')[0])
                              for name in batch_image_name]
        batch_image_data = preprocessing(batch_image_name, crop_size=img_size)

        yield (batch_image_data, batch_ground_truth)


# NOTE crop_size should be the same as h, w
# NOTE cv2.resize(image, (w, h))
# NOTE normalized by mo.py
# NOTE PIL return (width, height)
def preprocessing(image_name: list, crop_size: tuple) -> list:
    def center_crop(image):
        # image = cv2.resize(image, (crop_size[0]+32, crop_size[1]+32))
        top_left = [(image.size[0]-crop_size[0])//2,
                    (image.size[1]-crop_size[1])//2]

        return image.crop((top_left[0], top_left[1],
                          top_left[0]+crop_size[0], top_left[1]+crop_size[1]))

    image_batch = np.ndarray(
            shape=(len(image_name), 3, crop_size[0], crop_size[1]))
    for i, name in enumerate(image_name):
        # image_data = cv2.imread(name)[:, :, ::-1]
        image_data = Image.open(name).resize(
                (crop_size[0]+32, crop_size[1]+32))
        image_data = np.array(center_crop(image_data))
        image_data = image_data / 255.
        # Change data layout from HWC to CHW
        image_data = image_data.transpose((2, 0, 1))
        image_batch[i] = image_data

    return image_batch


def main(params):
    start_up_time = time.time()
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    model_xml = params['model']
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device
    # and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if params['cpu_extension'] and 'CPU' in params['device']:
        ie.add_extension(params['cpu_extension'], "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in params['device']:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [
                l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin "
                      "for specified device {}:\n {}".format(
                          params['device'], ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path "
                      "in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, \
        "Sample supports only single input topologies"
    assert len(net.outputs) == 1, \
        "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    # Define batch size here
    BATCH_SIZE = 64
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = BATCH_SIZE

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=params['device'])

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    data_iterator = data_loader(params['input'], n, (h, w))
    total_acc, total_correct, counter = 0, 0, 0
    inf_start = time.time()
    while True:
        try:
            images, labels = next(data_iterator)
            counter += len(labels)
            # FIXME fit last inconsistent batch
            if len(labels) != n:
                continue
            res = exec_net.infer(inputs={input_blob: images})

            # Processing output blob
            res = res[out_blob]
            pred = np.argsort(res)[:, -1]
            total_correct += np.sum(np.equal(pred, labels))
            total_acc = 100. * (total_correct / counter)
            print(f"Current accuracy is {total_acc:.2f}%, "
                  f"{total_correct}/{counter}", end="\r")
        except StopIteration:
            print(f"Run out of batch.")
            break
    inf_end = time.time()

    log.info(f"Startup loading time is "
             f"{(inf_start-start_up_time)*1000:.2f}ms")
    log.info(f"Average latency is "
             f"{(inf_end-inf_start)/counter*1000:.2f}ms")
    log.info(f"Total execution time is {inf_end-start_up_time:.2f}s")
    log.info(f"FPS is {counter/(inf_end-start_up_time):.2f}")
    log.info(f"FPS without startup time is {counter/(inf_end-inf_start):.2f}")

    return f"Over."


if __name__ == "__main__":
    parameters = build_argparser()
    print(main(parameters))
