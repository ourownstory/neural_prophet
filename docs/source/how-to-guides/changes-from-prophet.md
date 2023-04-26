# What has Changed from Prophet

NeuralProphet has a number of added features with respect to original Prophet.
They are as follows.

* Gradient Descent for optimisation via using PyTorch as the backend.
* Modelling Auto-Regression of time series using AR-Net
* Modelling lagged regressors using a separate linear or Feed-Forward Neural Network.
* Directly predict specific forecast horizons.
* Train a single model on many related time-series (global modelling).
* Flexible multiplicativity, one can set multiplicativity of future regressors and seasonality separately.

Due to the modularity of the code and the extensibility supported by PyTorch,
any component trainable by gradient descent can be added as a module
to NeuralProphet. Using PyTorch as the backend, makes the modelling process
much faster compared to original Prophet which uses Stan as the backend.
