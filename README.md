[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/ourownstory/neural_prophet?logo=github)](https://github.com/ourownstory/neural_prophet/releases)
[![Pypi_Version](https://img.shields.io/pypi/v/neuralprophet.svg)](https://pypi.python.org/pypi/neuralprophet)
[![Python Version](https://img.shields.io/badge/python-3.6+-blue?logo=python)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](https://opensource.org/licenses/MIT)

![NP-logo-wide_cut](https://user-images.githubusercontent.com/21246060/111388960-6c367e80-866d-11eb-91c1-46f2c0d21879.PNG)

Please note that the project is still in beta phase. Please report any issues you encounter or suggestions you have. We will do our best to address them quickly. Contributions are very welcome!

# NeuralProphet
A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch.

## Documentation
We are currently working on an improved [documentation page](http:/neuralprophet.com).

For a visual introduction to NeuralProphet, view the presentation given at the [40th International Symposium on Forecasting](notes/Presented_at_International_Symposium_on_Forecasting.pdf).

## Contribute
We compiled a [Contributing to NeuralProphet](CONTRIBUTING.md) page with practical instructions and further resources to help you become part of the family. 

## Community
#### Discussion and Help
If you have any question or suggestion, you can participate with [our community right here on Github](https://github.com/ourownstory/neural_prophet/discussions)

#### Slack Chat
We also have an active [Slack community](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg). Come and join the conversation!

## Tutorials
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ourownstory/neural_prophet)

There are several [example notebooks](https://github.com/ourownstory/neural_prophet/tree/master/example_notebooks) to help you get started.

Please refer to our [documentation page](https://ourownstory.github.io/neural_prophet/) for more resources.

### Minimal example
```python
from neuralprophet import NeuralProphet
```
After importing the package, you can use NeuralProphet in your code:
```python
m = NeuralProphet()
metrics = m.fit(df, freq="D")
future = m.make_future_dataframe(df, periods=30)
forecast = m.predict(future)
```
You can visualize your results with the inbuilt plotting functions:
```python
fig_forecast = m.plot(forecast)
fig_components = m.plot_components(forecast)
fig_model = m.plot_parameters()
```

## Install
You can now install neuralprophet directly with pip:
```shell
pip install neuralprophet
```
If you plan to use the package in a Jupyter notebook, we recommended to install the 'live' version:
```shell
pip install neuralprophet[live]
```
This will allow you to enable `plot_live_loss` in the `fit` function to get a live plot of train (and validation) loss.

If you would like the most up to date version, you can instead install direclty from github:
```shell
git clone <copied link from github>
cd neural_prophet
pip install .
```

## Model features
* Autocorrelation modelling through AR-Net
* Piecewise linear trend with optional automatic changepoint detection
* Fourier term Seasonality at different periods such as yearly, daily, weekly, hourly.
* Lagged regressors (measured features, e.g temperature sensor)
* Future regressors (in advance known features, e.g. temperature forecast)
* Holidays & special events
* Sparsity of coefficients through regularization
* Plotting for forecast components, model coefficients as well as final forecasts
* Automatic selection of training related hyperparameters

### Coming up soon
For details, please view the [Development Timeline](notes/development_timeline.md).

The next versions of NeuralProphet are expected to cover a set of new exciting features:

* Logistic growth for trend component.
* Uncertainty estimation of individual forecast components as well as the final forecasts. 
* Support for panel data by building global forecasting models.
* Incorporate time series featurization for improved forecast accuracy.
* Model bias modelling
* Unsupervised anomaly detection

For a complete list of all past and near-future changes, please refer to the [changelogs](changelogs.md).


## Authors
The project efford is led by Oskar Triebe (Stanford University), advised by Nikolay Laptev (Facebook, Inc) and Ram Rajagopal (Stanford University) and has been partially funded by Total S.A. The project has been developed in close collaboration with Hansika Hewamalage, who is advised by Christoph Bergmeir (Monash University). For a more complete list of contributors, please refer to the [contributors](contributors.md).

If you are interested in joining the project, please feel free to reach out to me (Oskar) - you can find my email on the [AR-Net Paper](https://arxiv.org/pdf/1911.12436.pdf).
