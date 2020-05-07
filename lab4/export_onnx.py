import onnx
import torch
import torchvision
import onnxruntime
import numpy as np


def main():
    # batch_size = 1
    num_cls = 11
    save_path = "./workspace/model.onnx"

    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_cls)
    model.load_state_dict(torch.load("./workspace/test_model.pt"))
    model.eval()
    print("Model loaded.")

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    # dynamic_axes={'input': {0: 'batch_size'},
    # 'output': {0: 'batch_size'}})
    torch.onnx.export(
            model, x, save_path, export_params=True,
            opset_version=10, do_constant_folding=True,
            input_names=['input'], output_names=['output'])
    print("Onnx model exported.")

    onnx_model = onnx.load("./workspace/model.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("./workspace/model.onnx")

    def to_numpy(tensor):
        return (tensor.detach().cpu().numpy()
                if tensor.requires_grad else tensor.cpu().numpy())

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
            to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime!")


if __name__ == "__main__":
    main()
