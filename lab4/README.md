# DLSR Homework4
- Using ResNet_V1_18 in all parts

## Lab4(a) Pruning with PyTorch
- Original training result(evaluation test) \
    ![](https://i.imgur.com/VwEmBAm.png)
- After **element-wise** pruning(global_sparsity >= 50% && acc_drop <= 10%)
    - Test result \
    ![](https://i.imgur.com/Ohsj7BI.png)
    - Global sparsity result \
    ![](https://i.imgur.com/uv9jbjs.png) ![](https://i.imgur.com/dLJWLYX.png)
- After **channel-wise** pruning(global_sparsity == 78.93% && acc == 80.55%)
    - Test result \
    ![](https://i.imgur.com/SUuTroP.png)
    - Global sparsity result \
    ![](https://i.imgur.com/nWlygLv.png) ![](https://i.imgur.com/RokFjNs.png)

## Lab4(b) OpenVINO Deployment
- My CPU information: \
    ![](https://i.imgur.com/xeKlJPe.png)
- Export your PyTorch model to ONNX, and compile it to IR with OpenVINO Model Optimizer \
    ![](https://i.imgur.com/WsS71Tx.png)
- Inference by OpenVINO inference engine, show accuracy, latency, and throughput of your model in PyTorch and OpenVINO respectively
    - Pytorch result \
        ![](https://i.imgur.com/lVzRewp.png)
    - OpenVINO result \
        ![](https://i.imgur.com/ks01Op1.png)
## Lab4(c\) Accuracy Checker Tool
- Show output accuracy from accuracy checker \
    ![](https://i.imgur.com/S5Zg5at.png)

## Lab4(d) Post-Training Optimization Tool
- DefaultQuantization result \
    ![](https://i.imgur.com/lXluQVC.png)
- AccuracyAwareQuantization result \
    ![](https://i.imgur.com/zhfifao.png)

## Questions
- Do you get any performance improvement in part(a) ? Why ?
    - No, because we can only do fake pruning in pytorch, which means that \
      sparse channels or elements are not remove from model truely. Modern \
      devices do not take advantage from spase computation. Consequently we \
      did not get performance improvement in part(a)
- Why can we get speedup using OpenVINO ?
    - We guess that OpenVINO has some optimization when running models. There \
      may be some optimization such as adaptive matrix speeding up, operation \
      fusing, data loading synchronization and so on.
- Please briefly explain how DefaultQuantization and AccuracyAwareQuantization works
    - DefaultQuantization
        1. Predefine range of activations of convolutional layers manually
        2. Automatically insert fakequant nodes into model graph based on hardware
        3. Initialize fakequant nodes according to calibration dataset
        4. Adjusts biases of convolutional adn fully-connected layers to make the overall error unbiased
    - AccuracyAwareQuantization
        1. Fully quantize model using DefaultQuantization algorithm
        2. Compare quantized model with original model, rank layers by target accuracy metric
        3. Based on the ranking, revert back some problematic layer to the original precision \
           until obtain pre-defined accuracy metrics.
        4. Note that AccuracyAwareQuantization may cause a degradation in performance in \
           comparison to DefaultQuantization, which is the same as our experimental \
           observation.
