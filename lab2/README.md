# Lab2 Introduction

- Test migration of imgaug and torch.transform: 
    - run `python visualize_test.py`
- Train under different data balance method: 
    - run `python train.py --data_dir=../food11re --balance=weighted(or augment)`
- Show distribution result after data balancing:
    - run `python my_dataset.py --data_dir=../food11re --balance=weighted(or augment)`
- Show evaluation result:
    - run `python eval_cls.py --data_dir=../food11re --balance=weighted(or augment)`
