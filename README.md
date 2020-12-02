[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NeuralProphet
A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net), built on PyTorch.

For a visual introduction to NeuralProphet, view the presentation given at the [40th International Symposium on Forecasting](notes/Presented_at_International_Symposium_on_Forecasting.pdf).

## Documentation
We are working on a [documentation page](https://ourownstory.github.io/neural_prophet/). Contributions welcome!

## Use 
### Install
After cloning the code repository  you can install neuralprophet as regular python package:
```shell
git clone <copied link from github>
pip install neural_prophet
```

If you plan to use the package in a Jupyter notebook, we recommended to install the 'live' version:
```shell
pip install neural_prophet[live]
```
This will allow you to enable `plot_live_loss` in the `fit` function to get a live plot of train (and validation) loss.

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
### Plotting
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
After downloading the code repository (via `git clone`), change to the repository directory (`cd neural_prophet`), activate your virtual environment, and install neuralprophet as python package with
`pip install -e .[dev]`

(Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.)

Additionally you must run `$ neuralprophet_dev_setup` in your console to run the dev-setup script which installs appropriate git hooks for Black and Unittests.

### Notes
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for doicstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo.

## Changelogs

### Coming up Next
The next versions of NeuralProphet are expected to cover a set of new exciting features:

* Robustify training for different datasets
* Logistic growth for trend component.
* Uncertainty estimation of individual forecast components as well as the final forecasts. 
* Support for panel data by building global forecasting models.
* Incorporate time series featurization for improved forecast accuracy.

### 0.2.6 (WIP)
* soft-start regularization
* confidence interval for forecast (as quantiles via pinball loss)

### 0.2.5 (current release)
* documentation pages added
* 1cycle policy
* learning rate range test
* tutorial notebooks: trend, events
* fixes to plotting, changepoints

## Development Timeline
For details, please view the [Development Timeline](notes/development_timeline.md).

## Authors
The alpha-stage NeuralProphet was developed by Oskar Triebe, advised by Ram Rajagopal (Stanford University) and Nikolay Laptev (Facebook, Inc), and was funded by Total S.A.
We are now further developing the beta-stage package in collaboration with Hansika Hewamalage, who is advised by Christoph Bergmeir (Monash University).
If you are interested in joining the project, please feel free to reach out to me (Oskar) - you can find my email on the [AR-Net Paper](https://arxiv.org/pdf/1911.12436.pdf).
