# TODOs

## General
* make all inputs 3D, including time.
* implement other frequency handling than daily. (Monthly, yearly and sub-daily data handling)
* fix bug when using get_last_forecasts(1) with n_lags=1
* document current model formulation in math (latex document)
* move verbose print statements to a logger


### PyTorch
* implement Learning-rate test 
* implement one-cycle learning rate schedule

### Autoregression
* visualize importance of each lag (overall)
* visualize value of each ar-weight for a given n-th forecast


### Seasonality
* Prophet documentation: 
https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
https://facebook.github.io/prophet/docs/multiplicative_seasonality.html
* regularize seasonality
* test on toy data
* implement alternative seasonality: month of year, day of week, hour of day, with varying smoothness

### Trend
* figure out how to stop gradients when delta-wise trend.
* make possible for user to set changepoint times.
* visualize important changepoint times

### Extra regressors
* adopt Prophet code for extra regressors 

### Events and Holidays
* TBD

## Future TODOs
* implement uncertainty estimation 
