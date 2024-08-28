import pytest

from neuralprophet import NeuralProphet


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
        model = NeuralProphet(**overrides)
        assert model.config_model.quantiles == expected


def test_config_training_quantiles_error_invalid_type():
    with pytest.raises(AssertionError) as err:
        _ = NeuralProphet(quantiles="hello world")
    assert str(err.value) == "Quantiles must be provided as list."


def test_config_training_quantiles_error_invalid_scale():
    with pytest.raises(Exception) as err:
        _ = NeuralProphet(quantiles=[-1])
    assert str(err.value) == "The quantiles specified need to be floats in-between (0, 1)."
    with pytest.raises(Exception) as err:
        _ = NeuralProphet(quantiles=[1.3])
    assert str(err.value) == "The quantiles specified need to be floats in-between (0, 1)."
