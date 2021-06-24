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

## Discussion and Help
[Discuss with our community here on Github](https://github.com/ourownstory/neural_prophet/discussions)

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


## Contribute
### Dev Install
Before starting it's a good idea to first create and activate a new virtual environment:
```python
python3 -m venv <path-to-new-env>
source <path-to-new-env>/bin/activate
```
Now you can install neuralprophet:

```python
git clone <copied link from github>
cd neural_prophet
pip install -e ".[dev]"
neuralprophet_dev_setup
git config pull.ff only 
```
Notes: 
* Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.
* The `neuralprophet_dev_setup` command runs the dev-setup script which installs appropriate git hooks for Black (pre-commit) and Unittests (pre-push).
* setting git to fast-forward only prevents accidental merges when using `git pull`.

### Style
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for docstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo.

### Slack Community
We have an active [Slack community](http://neuralprophet.slack.com/). Come and join the discussion!

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
