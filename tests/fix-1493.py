import datetime

import numpy as np
import pandas as pd

import neuralprophet.utils
from neuralprophet import NeuralProphet

rows = 1000

df = pd.DataFrame(
    {
        "ds": [datetime.datetime(2020, 1, 1) + datetime.timedelta(minutes=i) for i in range(rows)],
        "y": np.sin(2 * np.pi * np.arange(rows) / 100),
    }
)

model = NeuralProphet(n_lags=50, n_forecasts=5, epochs=2)

metrics = model.fit(df)
metrics = model.fit(df)

neuralprophet.utils.save(model, "repro.np")

future = model.make_future_dataframe(df, periods=100)
forecast = model.predict(future)

loaded_model = neuralprophet.utils.load("repro.np")

retrain_metrics = loaded_model.fit(df)  # error occurs here

print("Done.")
