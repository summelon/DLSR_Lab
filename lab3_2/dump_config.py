import json

config = {
        "config": {
            "experimentName": "ResNet_L3_TPE",
            "maxExecDuration": "1h",
            "maxTrialNum": 100,
            "parentProject": "None",
            "model": "train",
            "updatePeriod": 60,
            "tuner": {
                "builtinTunerName": "TPE",
                #"classArgs": {"optimize_mode": "maximize"}
            }
        },

        "params": {
            "save_model": True,
            "data_dir": "../food11re",
            "save_path": "./workspace/ckpt.pb",
            # "weighted" or "augment"
            "balance": "weighted",
            "layers": [3, 4, 23, 3],
            "resolution": 299,
            "width": 128

        },

        "search_space": {
            "batch_size": {
                "_type": "choice", "_value": [32, 64]},
            "lr": {
                "_type": "uniform", "_value": [1e-3, 1e-1]},
            "warmup_period": {
                "_type": "quniform", "_value": [5, 70, 1]},
            "num_epochs": {
                "_type": "quniform", "_value": [5, 70, 1]},
        }
    }

with open("./config.json", "w") as js:
    json.dump(config, js)
