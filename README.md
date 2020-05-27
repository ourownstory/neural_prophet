# bifrost

## TODOs

### General
* make all inputs 3D, including time.
* Monthly, yearly and sub-daily data handling

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
* implement alternative seasonality: month of year, day of week, hour of day

### Trend
* figure out how to stop gradients when delta-wise trend.
* make possible for user to set changepoint times.
* visualize important changepoint times

### Extra regressors
* adopt Prophet code for extra regressors 

### Events and Holidays
* TBD

## Future TODOs
* allow conditions for seasonal data
* Handle missing data
* implement uncertainty estimation 
