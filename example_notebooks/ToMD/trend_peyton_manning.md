# Fitting a changing trend

We will use the time series of the log daily page views for the Wikipedia page for Peyton Manning as an example to illustrate how to fit a changing trend. 

First, we load the data:

```python
if 'google.colab' in str(get_ipython()):
    !pip install git+https://github.com/ourownstory/neural_prophet.git # may take a while
    #!pip install neuralprophet # much faster, but may not have the latest upgrades/bugfixes
    data_location = "https://raw.githubusercontent.com/ourownstory/neural_prophet/master/"
else:
    data_location = "../"
```

```python
import pandas as pd
from neuralprophet import NeuralProphet
```

```python
df = pd.read_csv(data_location + "example_data/wp_log_peyton_manning.csv")
df.head(3)
```

ds | y | 
------------ | ------------- |
2007-12-10|9.59|
2007-12-11|8.52|
2007-12-12|8.18|

Now we can fit an initial model without any customizations.

We specify the data frequency to be daily. The model will remember this later when we predict into the future.

```python
m = NeuralProphet()
metrics = m.fit(df, freq="D")
metrics.head(3)
```

SmoothL1Loss | MAE | RegLoss |
------------ | ------------- |------------- |
1.292651|8.540894|0.0|
0.049019|1.122845|0.0|
0.010124|0.524999|0.0|

The returned metrics dataframe contains recoded metrics for each training epoch.

Next, we create a dataframe to predict on. 
Here, we specify that we want to predict one year into the future and that we want to include the entire history.

```python
future = m.make_future_dataframe(df, periods=365, n_historic_predictions=len(df))
future.tail(3)
```


ds|	y|	t|	y_scaled|
------------ | ------------- |------------- |------------- |
2017-01-17|NaN|1.122511|NaN|
2017-01-18|NaN|1.122848|NaN|
2017-01-19|NaN|1.123186|NaN|

Note: 'y' and 'y_scaled' are not given for the period extending into the future, as we do not know their true values.

```python
forecast = m.predict(future)
print(list(forecast.columns))
```

The returned forecast dataframe contains the original datestamps, 'y' values, the predicted 'yhat' values, residuals and all the individual model components.

```python
# plots the model predictions
fig1 = m.plot(forecast)
```

![trend_peyton_manning](plot/trend_peyton_manning_1.png){: style="height:350px"}

```python
# plots the individual forecast components for the given time period.
# fig = m.plot_components(forecast, residuals=True)

# visualizes the model parameters.
fig2 = m.plot_parameters()
```

![trend_peyton_manning](plot/trend_peyton_manning_2.png){: style="height:350px"}

# Adjusting Trend

The default values work fairly well in this example. However, the default of 5 changepoints may not be adequate if the actual change in trend happens to fall in a region between the points. 

## Increasing Trend Flexibility
We can address this by increasing the number of changepoints, giving the trend more flexibility, at the danger of overfitting.

Let's try what happens if we increase the number of changepoints to 30.
Additionally, we can increase the range of data on which we fit trend changepoints to only exlude the last 10 percent (default is 20 percent).

```python
m = NeuralProphet(
    n_changepoints=30,
    changepoints_range=0.90,    
)
metrics = m.fit(df, freq="D")
future = m.make_future_dataframe(df, n_historic_predictions=len(df))
forecast = m.predict(future)
```

```python
fig1 = m.plot(forecast)
fig2 = m.plot_parameters()
```
![trend_peyton_manning](plot/trend_peyton_manning_3.png){: style="height:350px"}

![trend_peyton_manning](plot/trend_peyton_manning_4.png){: style="height:350px"}

Looking at the trend rate changes it becomes evident that the trend is overfitting to short-term fluctuations.

## Automatic trendpoint selection
By adding regularization, we can achieve an automatic selection of the most relevant changepoints and draw the rate changes of other points close to zero. 

```python
m = NeuralProphet(
    n_changepoints=30,
    trend_reg=1.00,
    changepoints_range=0.90,    
)
metrics = m.fit(df, freq="D")
future = m.make_future_dataframe(df, n_historic_predictions=len(df))
forecast = m.predict(future)
```

```python
fig1 = m.plot(forecast)
fig2 = m.plot_parameters()
```
![trend_peyton_manning](plot/trend_peyton_manning_5.png){: style="height:350px"}

![trend_peyton_manning](plot/trend_peyton_manning_6.png){: style="height:350px"}

Now the model selects only a few relevant trend changepoints, drawing the rest closer to zero.

## Manual Trend Changepoints
You can also manually specify the trend changepoints.

Note: A changepoint will always be added at the beginning. You can ignore it.

```python
m = NeuralProphet(
    changepoints=['2012-01-01', '2014-01-01'],
)
metrics = m.fit(df, freq="D")
future = m.make_future_dataframe(df, n_historic_predictions=len(df))
forecast = m.predict(future)
```

```python
fig1 = m.plot(forecast)
fig2 = m.plot_parameters()
```
![trend_peyton_manning](plot/trend_peyton_manning_7.png){: style="height:350px"}

![trend_peyton_manning](plot/trend_peyton_manning_8.png){: style="height:350px"}

## Fine-tuning Trend Flexibility
We can adjust the regularization strength to get more or less points with a non-zero rate change.

Note: for too high regularization strengths, the model fitting process becomes unstable.

```python
m = NeuralProphet(
    n_changepoints=30,
    trend_reg=3.00,
    changepoints_range=0.90,   
)
metrics = m.fit(df, freq="D")
future = m.make_future_dataframe(df, n_historic_predictions=len(df))
forecast = m.predict(future)
```

```python
fig1 = m.plot(forecast)
fig2 = m.plot_parameters()
```
![trend_peyton_manning](plot/trend_peyton_manning_9.png){: style="height:350px"}

![trend_peyton_manning](plot/trend_peyton_manning_10.png){: style="height:350px"}

