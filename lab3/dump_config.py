import json

config = {
        "config": {
            "experimentName": "ResNet18_Random",
            "maxExecDuration": "1h",
            "maxTrialNum": 37,
            "parentProject": "None",
            "model": "train",
            "updatePeriod": 60,
            "tuner": {
                "builtinTunerName": "Random",
                #"classArgs": {"optimize_mode": "maximize"}
            }
        },

        "params": {
            "save_model": True,
            "data_dir": "../food11re",
            "save_path": "./workspace/ckpt.pb",
            # "weighted" or "augment"
            "balance": "weighted",

        },

        "search_space": {
            "batch_size": {
                "_type": "choice", "_value": [32, 64, 96, 128]},
            "lr": {
                "_type": "uniform", "_value": [1e-3, 1e-2]},
            "warmup_period": {
                "_type": "quniform", "_value": [5, 35, 1]},
            "num_epochs": {
                "_type": "quniform", "_value": [5, 25, 1]},
        }
    }

with open("./config.json", "w") as js:
    json.dump(config, js)
