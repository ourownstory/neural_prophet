from neuralprophet.configure import Train


def generate_config_train_params(overrides):
    config_train_params = {
        "quantiles": None,
        "learning_rate": None,
        "epochs": None,
        "batch_size": None,
        "loss_func": "Huber",
        "optimizer": "AdamW",
    }
    for key, value in overrides.items():
        config_train_params[key] = value
    return config_train_params


def test_config_training_quantiles():
    checks = [
        ({}, [0.5]),
        ({"quantiles": None}, [0.5]),
        ({"quantiles": []}, [0.5]),
        ({"quantiles": [0.2]}, [0.5, 0.2]),
        ({"quantiles": [0.2, 0.8]}, [0.5, 0.2, 0.8]),
        ({"quantiles": [0.5, 0.8]}, [0.5, 0.8]),
    ]

    for overrides, expected in checks:
        config_train_params = generate_config_train_params(overrides)
        config = Train(**config_train_params)
        assert config.quantiles == expected
