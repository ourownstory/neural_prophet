from neuralprophet.forecaster import NeuralProphet
import pandas as pd


class Prophet(NeuralProphet):
    def __init__(self, *args, **kwargs):
        # Set Prophet-like default args
        kwargs["quantiles"] = [0.9, 0.1]
        # Run the NeuralProphet function
        super(Prophet, self).__init__(*args, **kwargs)
        # Overwrite NeuralProphet properties
        self.name = "Prophet"
        self.history = None

    def fit(self, *args, **kwargs):
        # Run the NeuralProphet function
        metrics_df = super(Prophet, self).fit(*args, **kwargs)
        # Store the df for future use like in Prophet
        self.history = kwargs.get("df", args[0])
        return metrics_df

    def make_future_dataframe(self, *args, **kwargs):
        # Set Prophet-like default args
        kwargs["n_historic_predictions"] = kwargs.get("n_historic_predictions", True)
        # Use the provided df or fallback to the stored df during fit()
        try:
            df = kwargs.get("df", args[0])
        except:
            df = self.history
        kwargs["df"] = df
        # Run the NeuralProphet function
        df_future = super(Prophet, self).make_future_dataframe(**kwargs)
        return df_future
