## Development Timeline
### v0.1 Alpha [released in 2020]
Working version of NeuralProphet with missing features and potentially containing bugs.

### v0.2 to v0.5 Beta NeuralProphet [current]
Modelling capabilities:
* [done] Trend, piecewise linear changepoints
* [done] Auto-regression, univariate, multi-step ahead forecasts
* [done] Seasonalities, based on fourier-terms
* [done] Optional hidden layers for AR
* [done] Manage missing data - basic automatic imputation
* [done] Basic Automatic hyperparameter selection 
* [done] Custom Metrics
* [done] Training with evaluation on holdout set
* [done] Events and Holidays
* [done] Exagenous variables (as covariate inputs)
* Simple Uncertainty estimation

User Interface:
* simple package with limited capabilities
* similar to Facebook Prophet's basic features

Accompanying Products:
* Quickstart documentation and examples
* Benchmarks (Accuracy and Execution time)
* Few datasets

### v1.0 NeuralProphet
Added modelling capabilities:
* More intelligent Automatic hyperparameter selection
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

### v2.0 Redesigned - Modular Framework
Here, we will re-write large portions of the code structure in order to make it a modular framework where model components can freely be interchanged and combined. 

Added modelling capabilities:
* Inclusion of more potent models (Recurrence, Convolution, Attention, ...)

User Interface:
* Tools for Understanding of model and input-output mapping
* Integration with relevant Interfaces (Pytorch metrics, Tensorboard, scikitlearn, ...)

Accompanying Products:
* Pre-trained models

### v3.0 Nice UI for non-programmers
Alternative visual web-interface, potentially cloud-based execution
