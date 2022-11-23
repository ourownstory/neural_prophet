# Overview of the NeuralProphet Model

NeuralProphet is a Neural Network based PyTorch implementation of a user-friendly time series forecasting tool for practitioners. This is heavily inspired by [Prophet](https://facebook.github.io/prophet/), which is the popular forecasting tool developed by Facebook.

NeuralProphet is developed in a fully modular architecture which makes it scalable to add any additional components in the future. Our vision is to develop a simple to use forecasting tool for users while retaining the original objectives of Prophet such as interpretability, configurability and providing much more such as the automatic differencing capabilities by using PyTorch as the backend.

## Time Series Components

NeuralProphet is a decomposable time series model with the components, trend, seasonality, auto-regression, special events,
future regressors and lagged regressors. Future regressors are external variables which have known future values for the forecast
period whereas the lagged regressors are those external variables which only have values for the observed period. Trend can be
modelled either as a linear or a piece-wise linear trend by using changepoints. Seasonality is modelled using fourier terms and thus can handle multiple seasonalities for high-frequency data. Auto-regression is handled using an implementation of [AR-Net](https://github.com/ourownstory/AR-Net), an Auto-Regressive Feed-Forward Neural Network for time series.

Lagged regressors are also modelled using separate Feed-Forward Neural Networks. Future regressors and special events are both modelled as covariates of the model with dedicated coefficients. For more details, refer to the documentation of the individual components.

## Data Preprocessing

We perform a few data pre-processing steps in the model. For the observed values of the time series, users can specify whether they would like the values to be normalized. By default, the `y` values would be min-max normalized. If the user specifically, sets the `normalize_y` argument to `true`, the data is z-score normalized. Normalization can be performed for covariates as well. The default mode for normalization of covariates is `auto`. In this mode, apart from binary features such as events, all others are
z-score normalized.

We also perform an imputation in-case there are missing values in the data. However, imputation is only done if auto-regression is enabled in the model. In case of auto-regression, users may also choose not to impute any missing values and/or even drop missing values from the data, which should be done with caution as it may affect the model performance. Otherwise, the missing values do not really matter for the regression model. No special imputation is done for binary data. They are simply taken as `0` for the missing dates. For the numeric data, including the `y` values, normalization is a two-step process. First, small gaps are filled with a linear imputation and then the larger gaps are filled with rolling averages. When auto-regression is enabled, the observed `y` values are preprocessed in a moving window format to learn from lagged values. This is done for lagged regressors as well.

## When to Use NeuralProphet

NeuralProphet can produce both single step and multi step-ahead forecasts. NeuralProphet can build models based on a single time series or even from a group of time series. The latter is a recent addition to our forecasting tool widely known as global forecasting models.

NeuralProphet helps build forecasting models for scenarios where there are other external factors which can drive the behaviour of the target series over time. Using such external information can heavily improve forecasting models rather than relying only on the autocorrelation of the series. NeuralProphet tool is suitable for forecasting practitioners that wish to gain insights into the overall modelling process by visualizing the forecasts, the individual components as well as the underlying coefficients of the model. Through our descriptive plots, users can visualize the interaction of the individual components. They also have the power to control these coefficients as required by introducing sparsity through regularization. They can combine the components additively or multiplicatively as per their domain knowledge.

This is an ongoing effort. Therefore, NeuralProphet will be equipped with even much more features in the upcoming releases.
