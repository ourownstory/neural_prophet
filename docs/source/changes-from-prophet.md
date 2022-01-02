# What has Changed from Prophet

NeuralProphet has a number of added features with respect to original Prophet.
They are as follows.

* Gradient Descent for optimisation via using PyTorch as the backend.
* Modelling autocorrelation of time series using AR-Net
* Modelling lagged regressors using a sepearate Feed-Forward Neural Network.
* Configurable non-linear deep layers of the FFNNs.
* Tuneable to specific forecast horizons (greater than 1).
* Custom losses and metrics.

Due to the modularity of the code and the extensibility supported by PyTorch,
any component trainable by gradient descent can be added as a module
to NeuralProphet. Using PyTorch as the backend, makes the modelling process
much faster compared to original Prophet which uses Stan as the backend.