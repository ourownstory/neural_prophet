# Selecting the Hyperparameters

NeuralProphet has a number of hyperparameters that need to be specified by the user.
If not specified, default values for these hyperparameters will be used. 
View the `NeuralProphet` class in the API documentation of `forecaster.py` for details on all hyperparameters.

## Forecast horizon
`n_forecasts` is the size of the forecast horizon. 
The default value of 1 means that the model forecasts one step into the future. 

## Autoregression
`n_lags` defines whether the AR-Net is enabled (if `n_lags` > 0) or not.
The value for `n_lags` is usually recommended to be greater than `n_forecasts`, if possible
since it is preferable for the FFNNs to encounter at least `n_forecasts` length of the past
in order to predict `n_forecasts` into the future. Thus, `n_lags` determine how far into the 
past the auto-regressive dependencies should be considered. This could be a value chosen based
on either domain expertise or an empirical analysis.  

## Model Training Related Parameters
NeuralProphet is fit with stochastic gradient descent - more precisely, with an AdamW optimizer and a One-Cycle policy. 
If the parameter `learning_rate` is not specified, a learning rate range test is conducted to determine the optimal learning rate. 
The `epochs`, `loss_func` and `optimizer` are other parameters that directly affect the model training process. 
If not defined, `epochs`and `loss_func` are automatically set based on the dataset size. They are set in a manner that controls the total number training steps to be around 1000 to 4000.
NeuralProphet offers to set two different values for `optimizer`, namely `AdamW` and `SDG` (stochastic gradient descent).

If it looks like the model is overfitting to the training data (the live loss plot can be useful hereby), 
you can reduce `epochs`  and `learning_rate`, and potentially increase the `batch_size`. 
If it is underfitting, the number of `epochs` and `learning_rate` can be increased and the `batch_size` potentially decreased. 

The default loss function is the 'SmoothL1Loss' loss, which is considered to be robust to outliers. 
However, you are free to choose the standard `MSE` or any other PyTorch `torch.nn.modules.loss` loss function. 

## Increasing Depth of the Model
`ar_layers` defines the number of hidden layers and their sizes for the AR-Net in the overall model. It is an array where each element is the size of the corresponding hidden layer. The default is an empty array, meaning that the AR-Net will have only one final layer of size `n_forecasts`. Adding more layers results in increased complexity and also increased computational time, consequently. However, the added number of hidden layers can help build more complex relationships. To tradeoff between the computational complexity and the improved accuracy, the `ar_layers` is recommended to be set as an array with 1-2 elements. Nevertheless, in most cases, a good enough performance can be achieved by having no hidden layers at all.

`lagged_reg_layers` defines the number of hidden layers and their sizes for the lagged regressors' FFNN in the overall model. It is an array where each element is the size of the corresponding hidden layer. The default is an empty array, meaning that the FFNN of the lagged regressors will have only one final layer of size `n_forecasts`. Adding more layers results in increased complexity and also increased computational time, consequently. However, the added number of hidden layers can help build more complex relationships, especially useful for the lagged regressors. To tradeoff between the computational complexity and the improved accuracy, the `lagged_reg_layers` is recommended to be set as an array with 1-2 elements. Nevertheless, in most cases, a good enough performance can be achieved by having no hidden layers at all.

Please note that the previous `num_hidden_layers` and `d_hidden` arguments are now deprecated. The ar_net and covar_net architecture configuration is now input through `ar_layers` and `lagged_reg_layers`. If tuned manually, the recommended practice is to set values in between `n_lags` and `n_forecasts` for the sizes of the hidden layers. It is also important to note that with the current implementation, NeuralProphet allows you to specify different sizes for the hidden layers in both ar_net and covar_net.


## Data Preprocessing Related Parameters

`normalize_y` is about scaling the time series before modelling. By default, NeuralProphet performs a (soft) min-max normalization of the
time series. Normalization can help the model training process if the series values fluctuate heavily. However, if the series does 
not such scaling, users can turn this off or select another normalization. 

`impute_missing` is about imputing the missing values in a given series. 
Similar to Prophet, NeuralProphet too can work with missing values when it is in the regression mode without the AR-Net. 
However, when the autocorrelation needs to be captured, it is necessary for the missing values to be imputed, since then the modelling becomes an ordered problem. 
Letting this parameter at its default can get the job done perfectly in most cases.

`collect_metrics` is used to select the names of metrics to compute. It's set to `True` and computes `mae` and `rmse` by default.


## Trend Related Parameters
You can find a hands-on example at [`example_notebooks/trend_peyton_manning.ipynb`](https://github.com/ourownstory/neural_prophet/blob/master/example_notebooks/trend_peyton_manning.ipynb).

The trend flexibility if primarily controlled by `n_changepoints`, which sets the number of points where the trend rate may change.
Additionally, the trend rate changes can be regularized by setting `trend_reg` to a value greater zero.  
This is a useful feature that can be used to automatically detect relevant changepoints.

`changepoints_range` controls the range of training data used to fit the trend. 
The default value of 0.8 means that no changepoints are set in the last 20 percent of training data.

If a list of `changepoints` is supplied, `n_changepoints` and `changepoints_range`  are ignored. 
This list is instead used to set the dates at which the trend rate is allowed to change.

`n_changepoints` is the number of changepoints selected along the series for the trend. The default
value for this is 10.

## Seasonality Related Parameters
`yearly_seasonality`, `weekly_seasonality` and `daily_seasonality` are about which seasonal components to be modelled. For example, if you use temperature data, 
you can probably select daily and yearly. Using number of passengers using the subway would more likely have a weekly seasonality for example. 
Setting these seasonalities at the default `auto` mode, lets NeuralProphet decide which of them to include depending on how much data available. For example, the yearly seasonality will not
be considered if less than two years data available. In the same manner, the weekly seasonality will not be considered if less than two weeks available 
etc... However, if the user if certain that the series does not include yearly, weekly or daily seasonality, and thus the model should not be
distorted by such components, they can explicitly turn them off by setting the respective components to `False`. Apart from that, the parameters
`yearly_seasonality`, `weekly_seasonality` and `daily_seasonality` can also be set to number of Fourier terms of the respective seasonalities. 
The defaults are 6 for yearly, 4 for weekly and 6 for daily. Users can set this to any number they want. If the number of terms is 6 for yearly, that
effectively makes the total number of Fourier terms for the yearly seasonality 12 (6*2), to accommodate both sine and cosine terms.
Increasing the number of Fourier terms can make the model capable of capturing quite complex seasonal patterns. However, similar to the `ar_layers`,
this too results in added model complexity. Users can get some insights about the optimal number of Fourier terms by looking at the final component
plots. The default `seasonality_mode` is additive. This means that no heteroscedasticity is expected in the series in terms of the seasonality. 
However, if the series contains clear variance, where the seasonal fluctuations become larger proportional to the trend, the `seasonality_mode`
can be set to multiplicative.

## Regularization Related Parameters
NeuralProphet also contains a number of regularization parameters to control the model coefficients and introduce sparsity into the model. This also
helps avoid overfitting of the model to the training data. For `seasonality_reg`, small values in the range 0.1-1 allow to fit large seasonal 
fluctuations whereas large values in the range 1-100 impose a heavier penalty on the Fourier coefficients and thus dampens the seasonality. 
For `ar_sparsity` values in the range 0-1 are expected with 0 inducing complete sparsity and 1 imposing no regularization at all. `ar_sparsity` along with
 `n_lags` can be used for data exploration and feature selection. You can use a larger number of lags thanks to the scalability of AR-Net and use the scarcity 
 to identify important influence of past time steps on the prediction accuracy. For `future_regressor_regularization`, `event_regularization` and `country_holiday_regularization`, values can be set in between 0-1 in the same notion
as in `ar_sparsity`. You can set different regularization parameters for the individual regressors and events depending on which ones need to be more
dampened.