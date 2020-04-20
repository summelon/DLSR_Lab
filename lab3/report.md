# DLSR Homework3
- Source code: `https://github.com/summelon/DLSR_Lab/tree/master/lab3`
- Using `ResNet_V1_18` in Lab3-1(a),(b),(d)
- Using `ResNet_V1_50` as teacher, `MobileNet_V2_224_1.0` as student in (c\)
## Lab 3-1(a) Early Stopping
1. Q: compare the training with/without applying the `EarlyStopping`
- Training with pretrain model
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
    - Training without pretrain model
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

## Lab 3-1(c\) Knowledge Distillation
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
- Analysis between tuners
    - Result(highest accuracy)
        - TPE: 87.1%
        - Random: 86.6%    
    - The moset impact parameter
        - TPE: `warmup_period`
        - Random: `batch_size`
        - Note that `learning_rate` is the second impact parameter
    - Conclusion
        - `warmup_period` is the most important parameter for `TPE` 
          tuner to search and `batch_size` is the trivial one. On the 
          contrary, in `Random` tuner, situation is totally different,
          i.e. `batch_size` is the most important one. 
        - In my opinion, `warmup_period`, which is the crucial parameter
          in warm up start, is the key point of the training step. Maybe
          because of this, `TPE` is able to overperform `Random` slightly.
- Compare `my setting` to `Dr.Opt's`
    - Compare to my previous training without `Dr.Opt`, accuracy is 
      significantly improved from 85%(mine) to 87%(Dr.Opt's), which is 2%.
        - My training setting
            - batch_size: 64
            - lr: 5.6e-3
            - num_epochs: 14
            - warmup_period: 34
        - Dr.Opt's best
            - batch_size: 64
            - lr: 9e-3
            - num_epochs: 17
            - warmup_period: 27
    - Conclusion
        - My setting is approximate to Dr.Opt's in each parameter. While
          Dr.Opt is significantly higher than mine. I think Dr.Opt find deeper
          relationship between `num_epochs` and `warmup_period` than me. Their
          correlation is so important that it determines the overview learning
          rate schedule. Addtionally, a fitter destination of `lr` is also
          helpful for higher accuracy. Dr.Opt with better algorithms may find
          reasonable range of `lr` easier.



