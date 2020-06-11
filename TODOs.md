# TODOs

## General
* make all inputs 3D, including time.
* implement other frequency handling than daily. (Monthly, yearly and sub-daily data handling)
* fix bug when using get_last_forecasts(1) with n_lags=1
* document current model formulation in math (latex document)
* move verbose print statements to a logger


## PyTorch
* implement Learning-rate test 
* implement one-cycle learning rate schedule


## Seasonality
* regularize seasonality
* test intra-day seasonality

## Trend
* Stop gradients when delta-wise trend.
* make possible for user to set changepoint times.


## Extra regressors
* To be started.

## Events and Holidays
* To be started. Prophet documentation: 
https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html

## Uncertainty
* To be started. implement basic uncertainty estimation via Quantile Regression


## Future TODOs
* Measure computation times, Find bottlenecks and speed up.
* Trend: Better changepoint detection, visualize important changepoints
* Add support for data without datestamps
