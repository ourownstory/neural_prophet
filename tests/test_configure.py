from neuralprophet.configure import Train


config_train_defaults = {
    "quantiles": None,
    "learning_rate": None,
    "epochs": None,
    "batch_size": None,
    "loss_func": "Huber",
    "optimizer": "AdamW",
}


def test_config_training_quantiles_initialization_none():
    train = Train(**config_train_defaults, quantiles=None)
    assert train.quantiles == [0.5]


def test_config_training_quantiles_initialization_empty():
    train = Train(**config_train_defaults, quantiles=[])
    assert train.quantiles == [0.5]
