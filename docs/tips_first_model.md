# Tips to select hyperparameter for your first model

`n_forecast` should be user input and defined by your use case. It refers to the number of time step ahead the model will forecast.

`yearly_seasonality`, `weekly_seasonality`, `daily_seasonality` can be left to "auto", otherwise if you know your data has a seasonality you change it to True. For example, if you use temperature data, you can probably select daily and yearly. Using number of passengers using the subway would more likely have a weekly seasonality for example.

`n_lags`, and `ar_sparsity` are important parameters in NeuralProphet.  `n_lags` refers to the number of past time step to consider as features for the prediction, `ar_sparsity` the lower the more sparsity. These two parameters can be used for data exploration and feature selection. You can use a larger number of lags thanks to the scalability of AR-Net and use the scarcity to identify important influence of past time steps on the prediction accuracy. 

`num_hidden_layers` and   `d_hidden` are neural network parameters. `num_hidden_layers` refers to the number of hidden layers and `d_hidden` refers to the number of neurons per hidden layer. You can usually start with no hidden layer to start exploring your data and test the AR feature selection and then add layers. You can first try with 1-2 layers and 10-20 neurons.

You can also use the `add_lagged_regressor` function to add more features to your model. NeuralProphet allows you to add features and then look at AR relevance of each past time step of each feature. For example, if you want to predict temperature, you might have access to wind or humidity data. Adding those data as a lagged regressor will let the model use as many past time steps as it considers for the target data. It can help decide if a specific feature is well a useful predictor.
