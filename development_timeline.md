## Development Timeline
###v0.1
Modelling capabilities:
* Trend, piecewise linear changepoints
* Auto-regression, univariate, multi-step ahead forecasts
* Seasonalities, piecewise linear
* Events and Holidays
* Exagenous variables (as covariate inputs)
* Optional hidden layers for AR
* Simple Uncertainty estimation

User Interface:
* simple package with limited capabilities
* similar to Prophet's basic features

Accompanying Products:
* Quickstart documentation and examples
* Benchmarks (Accuracy and Execution time)

### v1.0
Added modelling capabilities:
* Manage missing data
* Automatic hyperparameter selection (make hyperparameters optional)
* different ways to manage trend/normalize data and compute seasonality (rolling, local seasonality, ...)
* Inclusion of traditional models (ets, sarimax, ...)
* Component-wise uncertainty

User Interface:
* More user-control (set trend changepoint times, ...)
* Rich analytics and plotting 
* Model gives user feedback on how to improve hyperparameters (if set)
* Integration with Time-Series Preprocessing tools

Accompanying Products:
* Large collection of time-series datasets
* Professional documentation and more tutorials

### v2.0
Added modelling capabilities:
* Inclusion of more potent models (Recurrence, Convolution, Attention, ...)

User Interface:
* Tools for Understanding of model and input-output mapping
* Integration with relevant Interfaces (Pytorch metrics, Tensorboard, scikitlearn, ...)

Accompanying Products:
* Pre-trained models

### v3.0
Added modelling capabilities:
* TBD

User Interface:
* TBD

Accompanying Products:
* Alternative visual web-interface, potentially cloud-based execution
