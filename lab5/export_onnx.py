import onnx
import torch
import onnxruntime
import numpy as np
import model as lab5_model


def main():
    # batch_size = 1
    num_cls = 11
    save_path = "./workspace/special_model.onnx"

    model = lab5_model.SpecialModel()
    # Retore pretrained weight
    pretrained_dict = torch.load("./specialmodel_pretrain.pth")
    model_dict = model.state_dict()
    pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    num_ftrs = model.fc2.in_features
    model.fc2 = torch.nn.Linear(num_ftrs, num_cls)
    model.load_state_dict(torch.load("./workspace/model.pt"))
    model.eval()
    print("Model loaded.")

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    # dynamic_axes={'input': {0: 'batch_size'},
    # 'output': {0: 'batch_size'}})
    torch.onnx.export(
            model, x, save_path, export_params=True,
            opset_version=10, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'bs'}})
    print("Onnx model exported.")

    onnx_model = onnx.load("./workspace/special_model.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("./workspace/special_model.onnx")

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
