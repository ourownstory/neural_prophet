from neuralprophet.configure import Train


config_train_defaults = {}
config_train_defaults.setdefault("quantiles", None)
config_train_defaults.setdefault("learning_rate", None)
config_train_defaults.setdefault("epochs", None)
config_train_defaults.setdefault("batch_size", None)
config_train_defaults.setdefault("loss_func", "Huber")
config_train_defaults.setdefault("optimizer", "AdamW")


def test_config_training_quantiles_initialization_none():
    params = config_train_defaults.copy()
    params["quantiles"] = None
    train = Train(**params)
    assert train.quantiles == [0.5]


def test_config_training_quantiles_initialization_empty():
    params = config_train_defaults.copy()
    params["quantiles"] = []
    train = Train(**params)
    assert train.quantiles == [0.5]
