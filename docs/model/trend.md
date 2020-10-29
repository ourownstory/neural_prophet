# Trend

## Default Trend Modelling:

Direct gradient, 5 changepoints
```python
m = NeuralProphet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)
metrics
```


Forecasting
```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_comp = m.plot_components(forecast)
fig_param = m.plot_parameters()
```

## Automatic sparse changepoint detection: 
2x regularized trend, 100 changepoints

```python
m = NeuralProphet(
    n_changepoints=100,
    trend_smoothness=2,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)

```

```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_param = m.plot_parameters()

```

## Underfit: no changepoints

```python
m = NeuralProphet(
    n_changepoints=0,
    trend_smoothness=0,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df, validate_each_epoch=True)
```

```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_param = m.plot_parameters()
```

## Smooth Underfit: 10x regularized trend, 100 changepoints

```python
m = NeuralProphet(
    n_changepoints=100,
    trend_smoothness=5,
    trend_threshold=True,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)
```
```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_param = m.plot_parameters()

```

## Overfit: Direct gradient, 100 changepoints

```python
m = NeuralProphet(
    n_changepoints=100,
    trend_smoothness=0,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)
```

```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_param = m.plot_parameters()
```

## Overfit: Direct gradient, discontinuous trend, 100 changepoints

```python
m = NeuralProphet(
    n_changepoints=100,
    trend_smoothness=-1,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)

```

```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_param = m.plot_parameters()
```

## Smooth overfit: 0.1xregularized trend, 100 changepoints

```python
m = NeuralProphet(
    n_changepoints=100,
    trend_smoothness=0.1,
    trend_threshold=True,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)

```

```python
future = m.make_future_dataframe(df, future_periods=365, n_historic_predictions=len(df))
forecast = m.predict(future)
fig_fit = m.plot(forecast)
fig_param = m.plot_parameters()

```

