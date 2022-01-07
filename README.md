[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/ourownstory/neural_prophet?logo=github)](https://github.com/ourownstory/neural_prophet/releases)
[![Pypi_Version](https://img.shields.io/pypi/v/neuralprophet.svg)](https://pypi.python.org/pypi/neuralprophet)
[![Python Version](https://img.shields.io/badge/python-3.6+-blue?logo=python)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ourownstory/neural_prophet/actions/workflows/ci.yml/badge.svg)](https://github.com/ourownstory/neural_prophet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ourownstory/neural_prophet/branch/master/graph/badge.svg?token=U5KXCL55DW)](https://codecov.io/gh/ourownstory/neural_prophet)
[![Slack](https://img.shields.io/badge/slack-@neuralprophet-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40)](https://neuralprophet.slack.com/join/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg#/shared-invite/email)
[![Downloads](https://static.pepy.tech/personalized-badge/neuralprophet?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/neuralprophet)

![NP-logo-wide_cut](https://user-images.githubusercontent.com/21246060/111388960-6c367e80-866d-11eb-91c1-46f2c0d21879.PNG)


Please note that the project is still in beta phase. Please report any issues you encounter or suggestions you have. We will do our best to address them quickly. Contributions are very welcome!

# NeuralProphet
A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch.

## Documentation
The [documentation page](https://neuralprophet.com) may not we entirely up to date. Docstrings should be reliable, please refer to those when in doubt. We are working on an improved documentation. We appreciate any help to improve and update the docs.

For a visual introduction to NeuralProphet, [view this presentation](notes/NeuralProphet_Introduction.pdf).

## Contribute
We compiled a [Contributing to NeuralProphet](CONTRIBUTING.md) page with practical instructions and further resources to help you become part of the family. 

## Community
#### Discussion and Help
If you have any question or suggestion, you can participate with [our community right here on Github](https://github.com/ourownstory/neural_prophet/discussions)

#### Slack Chat
We also have an active [Slack community](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg). Come and join the conversation!

## Tutorials
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ourownstory/neural_prophet)

There are several [example notebooks](tutorials/) to help you get started. 

You can find the datasets used in the tutorials, including data preprocessing examples, in our [neuralprophet-data repository](https://github.com/ourownstory/neuralprophet-data).

Please refer to our [documentation page](https://neuralprophet.com) for more resources.

### Minimal example
```python
from neuralprophet import NeuralProphet
```
After importing the package, you can use NeuralProphet in your code:
```python
m = NeuralProphet()
metrics = m.fit(df, freq="D")
forecast = m.predict(df)
```
You can visualize your results with the inbuilt plotting functions:
```python
fig_forecast = m.plot(forecast)
fig_components = m.plot_components(forecast)
fig_model = m.plot_parameters()
```
If you want to forecast into the unknown future, extend the dataframe before predicting:
```python
m = NeuralProphet().fit(df, freq="D")
df_future = m.make_future_dataframe(df, periods=30)
forecast = m.predict(df_future)
fig_forecast = m.plot(forecast)
```
## Install
You can now install neuralprophet directly with pip:
```shell
pip install neuralprophet
```

### Install options

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
* Country holidays & recurring special events
* Sparsity of coefficients through regularization
* Plotting for forecast components, model coefficients as well as final predictions
* Automatic selection of training related hyperparameters
* Support for panel data by building global forecasting models.

### Coming up soon
For details, please view the [Development Timeline](notes/development_timeline.md).

The next versions of NeuralProphet are expected to cover a set of new exciting features:

* Logistic growth for trend component.
* Uncertainty estimation of predicted values
* Incorporate time series featurization for improved forecast accuracy.
* Model bias modelling/correction with secondary model
* Multimodal dynamics: unsupervised automatic modality-specific forecast.

For a complete list of all past and near-future changes, please refer to the [changelogs](notes/changelogs.md).

## Cite
Please cite [NeuralProphet](https://arxiv.org/abs/2111.15397) in your publications if it helps your research:
```
@misc{triebe2021neuralprophet,
      title={NeuralProphet: Explainable Forecasting at Scale}, 
      author={Oskar Triebe and Hansika Hewamalage and Polina Pilyugina and Nikolay Laptev and Christoph Bergmeir and Ram Rajagopal},
      year={2021},
      eprint={2111.15397},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Authors
The project effort is led by Oskar Triebe (Stanford University), advised by Nikolay Laptev (Facebook, Inc) and Ram Rajagopal (Stanford University) and has been partially funded by Total S.A. The project has been developed in close collaboration with Hansika Hewamalage, who is advised by Christoph Bergmeir (Monash University). The project has evolved into a community project, supported by people like you. 

If you are interested in joining the project, please feel free to reach out to me (Oskar) - you can find my email on the [AR-Net Paper](https://arxiv.org/pdf/1911.12436.pdf).
