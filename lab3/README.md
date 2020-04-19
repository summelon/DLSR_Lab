# DLSR Homework3
- Using `ResNet V1 18` in Lab3-1(a),(b),(d)
## Lab 3-1(a) Early Stopping
1. Q: compare the training with/without applying the `EarlyStopping`
- Training with pretrain model
![](https://i.imgur.com/dYhj9Sh.png)
    - Use `EarlyStopping`
        - Training stopped at 13th epoch
        - Accuracy is 82.97%
    - Without `EarlyStopping`
        - training stopped at 17th epoch
        - Accuracy is 83.03%
- Conclusion
    - Using `EarlyStopping` avoids redundant training time significantly
    - Training without `EarlyStopping` might lead to a little bit higher accuracy

## Lab 3-1(b) Transfer Learning
1. Q: Compare the training (speed of convergence, accuracy, etc.)
   w/wo transfer learning
    - Training without pretrain model\
    ![](https://i.imgur.com/qGjaf9A.png =50%x)
    - Comparision
        - Use `pretrain model`
            - Converged in 11th epoch
            - Accuracy: 82.97%
        - Without `pretrain model`
            - Converged in 12th epoch
            - Accuracy: 48.3%
    
2. Q: How to adjust the learning rate after applying transfer learning?\
   Please do experiments for different learning rate settings
    - Apllied `warmup_start` + `cos_annealing_lr` to adjust learning rate
        - Before
            - Learning rate = 4e-3
            - Converged in 11th epoch
            - Accuracy: 82.97%
        - After
            - Additional setting
                - T_max in cos_annealing_lr: 
                  $T_{max}=length_{dataset} // BatchSize*N_{epoch}$ 
                - Warmup_period = 25
            - Learning rate = 1e-5 -> 4e-9 -> 0
            - Run out of all epoches
            - Accuracy: 85.07%
3. Conclusion: transfer learning is very helpful to imporve both the
   speed of convergence and accuracy

## Lab 3-1(c) Knowledge Distillation
1. Q: Compare the training progress (speed, accuracy, etc.)\
   of the student model w/wo applying knowledge distillation
- Teacher model: `ResNet_V1_50`, student model `MobileNet_V2`
    - λ = 0.95
    - Original training accuracy
        - Teacher model: 85.77%
        - Student model: 81.80%
- Use `Knowledge Distillation`
    | K/Student Model | Accuracy |
    |        -------- | -------- |
    |           K = 2 | 84.87%   |
    |           K = 4 | 82.97%   |
    |           K = 6 | 80.46%   |
- Conclusion: Using `Knowledge Distillation` performs better than
  original training on student model. It shows both in significant
  improvement of accuracy and faster speed of convergence. Since I used
  `warmup start` and `cos_anneal_lr`, program run out of all epochs.
  However it's obvious that speed of convergence when `K=2` is faster.

## Lab 3-1(d) Hyper-parameter Search
- Dr.Opt setting
    - Tunner: TPE, Random
    - Search space
        - "batch_size":
            "_type": "choice", "_value": [32, 64, 96, 128]
        - "lr": 
            "_type": "uniform", "_value": [1e-3, 1e-2]
        - warmup_period":
            "_type": "quniform", "_value": [5, 35, 1]},
        - "num_epochs":
            "_type": "quniform", "_value": [5, 25, 1]
- Analysis
    - Impact parameter
