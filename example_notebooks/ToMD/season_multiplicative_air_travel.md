# Multiplicative Seasonality 
By default NeuralProphet fits additive seasonalities, meaning the effect of the seasonality is added to the trend to get the forecast. This time series of the number of air passengers is an example of when additive seasonality does not work:

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
from neuralprophet import NeuralProphet, set_log_level
# set_log_level("ERROR")
```

```python
m = NeuralProphet()
df = pd.read_csv(data_location + "example_data/air_passengers.csv")
metrics = m.fit(df, freq="MS")
```

```python
future = m.make_future_dataframe(df, periods=50, n_historic_predictions=len(df))
forecast = m.predict(future)
fig = m.plot(forecast)
# fig_param = m.plot_parameters()
```

![season_multiplicative_air_travel](plot/season_multiplicative_1.png){: style="height:350px"}

This time series has a clear yearly cycle, but the seasonality in the forecast is too large at the start of the time series and too small at the end. In this time series, the seasonality is not a constant additive factor as assumed by NeuralProphet, rather it grows with the trend. This is multiplicative seasonality.

NeuralProphet can model multiplicative seasonality by setting `seasonality_mode="multiplicative"` in the input arguments:

```python
m = NeuralProphet(seasonality_mode="multiplicative")
metrics = m.fit(df, freq="MS")
```

```python
future = m.make_future_dataframe(df, periods=50, n_historic_predictions=len(df))
forecast = m.predict(future)
fig = m.plot(forecast)
# fig_param = m.plot_parameters()
```

![season_multiplicative_air_travel](plot/season_multiplicative_2.png){: style="height:350px"}

The components figure will now show the seasonality as a percent of the trend:

```python
fig_param = m.plot_components(forecast)
```

![season_multiplicative_air_travel](plot/season_multiplicative_3.png){: style="height:350px"}

Note that the seasonality is only fit on data occuring at the start of the month. Thus, the plotted values for seasonality inbetween months may take on random values. 

Setting`seasonality_mode="multiplicative"` will model all seasonalities as multiplicative, including custom seasonalities added with `add_seasonality`.
