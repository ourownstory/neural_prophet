import time
import pandas as pd
import psutil
from neuralprophet import NeuralProphet


data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"
file = "energy/SF_hospital_load.csv"

data_df = pd.read_csv(data_location + file)
data_df.shape

rng_2x = pd.date_range(start='2015-01-01 01:00:00', freq='H', periods=2*8760)
y_2x = pd.concat([data_df['y'], data_df['y']], ignore_index=True)
data_df_2x = pd.DataFrame({'ds': rng_2x, 'y': y_2x})
data_df_2x.tail(5)

rng_10x = pd.date_range(start='2015-01-01 01:00:00', freq='H', periods=10*8760)
y_10x = pd.concat([data_df_2x['y'], data_df_2x['y'], data_df_2x['y'], data_df_2x['y'], data_df_2x['y']], ignore_index=True)
data_df_10x = pd.DataFrame({'ds': rng_10x, 'y': y_10x})
data_df_10x.shape

quantile_lo, quantile_hi = 0.05, 0.95
quantiles = [quantile_lo, quantile_hi]

m = NeuralProphet(
    epochs=1,
    batch_size=128,
    learning_rate=1.0,
    n_forecasts=7,
    n_lags=14,
    quantiles=quantiles
)

m2x = NeuralProphet(
    epochs=1,
    batch_size=128,
    learning_rate=1.0,
    n_forecasts=7,
    n_lags=14,
    quantiles=quantiles
)

m10x = NeuralProphet(
    epochs=1,
    batch_size=128,
    learning_rate=1.0,
    n_forecasts=7,
    n_lags=14,
    quantiles=quantiles
)

metrics = m.fit(data_df, freq="H")
metrics2x = m2x.fit(data_df_2x, freq="H")
metrics10x = m10x.fit(data_df_10x, freq="H")

forecast = m.predict(data_df)
forecast2x = m2x.predict(data_df_2x)
forecast10x = m10x.predict(data_df_10x)

######## Test 1 ###########
# get the start time
st = time.time()

fig1 = m.highlight_nth_step_ahead_of_each_forecast(1).plot(
    forecast, plotting_backend='plotly').show()

# get the end time
et = time.time()
elapsed_time = et - st
print('Execution time test1:', elapsed_time, 'seconds')
print('RAM memory % used:', psutil.virtual_memory()[2])
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

######## Test 2 ###########
# get the start time
st = time.time()

fig2 = m10x.highlight_nth_step_ahead_of_each_forecast(1).plot(
    forecast10x, plotting_backend='plotly').show()

# get the end time
et = time.time()
elapsed_time = et - st
print('Execution time test2:', elapsed_time, 'seconds')
print('RAM memory % used:', psutil.virtual_memory()[2])
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


