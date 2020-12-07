[![Pypi_Version](https://img.shields.io/pypi/v/neuralprophet.svg)](https://pypi.python.org/pypi/neuralprophet)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Please note that the project is still in beta phase. Please report any issues you encounter or suggestions you have. We will do our best to address them quickly. Contributions are also highly welcome!

# NeuralProphet
A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch.

For a visual introduction to NeuralProphet, view the presentation given at the [40th International Symposium on Forecasting](notes/Presented_at_International_Symposium_on_Forecasting.pdf).

## Documentation
We are working on a [documentation page](https://ourownstory.github.io/neural_prophet/). Contributions welcome!

## Use 

### Install
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
Please note that NeuralProphet requires a Python version >= 3.7 due to the use of `@dataclass`.
### Basic example
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


## Model Features
* Autocorrelation modelling through AR-Net
* Piecewise linear trend
* Fourier term Seasonality at different periods such as yearly, daily, weekly, hourly.
* Lagged regressors
* Future regressors
* Holidays & special events
* Sparsity of coefficients through regularization
* Plotting for forecast components, model coefficients as well as final forecasts

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
pip install -e .[dev]
neuralprophet_dev_setup
```
Notes: 
* The last command runs the dev-setup script which installs appropriate git hooks for Black (pre-commit) and Unittests (pre-push).
* Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.


### Style
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for docstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo.

## Changelogs
### Coming up soon
For details, please view the [Development Timeline](notes/development_timeline.md).

The next versions of NeuralProphet are expected to cover a set of new exciting features:

* Robustify training for different datasets
* Logistic growth for trend component.
* Uncertainty estimation of individual forecast components as well as the final forecasts. 
* Support for panel data by building global forecasting models.
* Incorporate time series featurization for improved forecast accuracy.

### 0.2.8 (next)
* soft-start regularization
* confidence interval for forecast (as quantiles via pinball loss)

### 0.2.7 (current)
* example notebooks: Sub-daily data, Autoregresseion
* bugfixes: `lambda_delay`, `train_speed`

### 0.2.6 
* Auto-set `batch_size` and `epochs`
* add `train_speed` setting
* add `set_random_seed` util
* continued removal of `AttrDict` uses
* bugfix to index issue in `make_future_dataframe`

### 0.2.5
* documentation pages added
* 1cycle policy
* learning rate range test
* tutorial notebooks: trend, events
* fixes to plotting, changepoints

## Authors
The project efford is led by Oskar Triebe (Stanford University), advised by Nikolay Laptev (Facebook, Inc) and Ram Rajagopal (Stanford University) and has been partially funded by Total S.A. The project has been developed in close collaboration with Hansika Hewamalage, who is advised by Christoph Bergmeir (Monash University).

### Contributors
This is the list of NeuralProphet's significant contributors.
This does not necessarily list everyone who has contributed code.
To see the full list of contributors, see the revision history in source control.
* Oskar Triebe
* Hansika Hewamalage
* Nikolay Laptev
* Riley Dehaan
* Gonzague Henri
* Ram Rajagopal
* Christoph Bergmeir
* Italo Lima
* Caner Komurlu
* Rodrigo Riveraca


If you are interested in joining the project, please feel free to reach out to me (Oskar) - you can find my email on the [AR-Net Paper](https://arxiv.org/pdf/1911.12436.pdf).
