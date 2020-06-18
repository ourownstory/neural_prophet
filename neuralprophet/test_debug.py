import pandas as pd
from neuralprophet.neural_prophet import NeuralProphet
import matplotlib.pyplot as plt


def test_eval(verbose=True):
    from neuralprophet.df_utils import split_df
    # df = pd.read_csv('../data/example_air_passengers.csv')
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # df.head()
    # print(df.shape)
    m = NeuralProphet(
        n_lags=14,
        n_forecasts=7,
        verbose=verbose,
        ar_sparsity=0.1,
        loss_func='huber',
    )

    df_train, df_val = m.split_df(df, valid_p=0.2)
    train_metrics = m.fit(df_train)
    val_metrics = m.test(df_val)
    # if verbose:
    #     print("Train Metrics:")
    #     print(train_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))
    #     print("Val Metrics:")
    #     print(val_metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))

    # metrics = m.fit(df, test_each_epoch=True, valid_p=0.2)
    # if verbose:
    #     print(metrics.to_string(float_format=lambda x: "{:6.3f}".format(x)))


def test_trend():
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        n_changepoints=100,
        trend_smoothness=0,
        trend_threshold=True,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        loss_func='huber',
        verbose=True,
    )
    # m.set_forecast_in_focus(m.n_forecasts)
    m.fit(df)
    forecast = m.predict(future_periods=60)
    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()


def test_ar_net(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    m = NeuralProphet(
        verbose=verbose,
        n_forecasts=3,
        n_lags=10,
        n_changepoints=0,
        trend_smoothness=0,
        # ar_sparsity=0.1,
        num_hidden_layers=0,
        # num_hidden_layers=2,
        d_hidden=64,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.set_forecast_in_focus(m.n_forecasts)
    m.fit(df)
    forecast = m.predict()
    if verbose:
        # m.plot_last_forecasts(3)
        # m.plot(forecast)
        # m.plot(forecast, crop_last_n=10+m.n_lags+m.n_forecasts)
        m.plot_components(forecast)
        plt.show()


def test_seasons(verbose=True):
    df = pd.read_csv('../data/example_wp_log_peyton_manning.csv')
    # m = NeuralProphet(n_lags=60, n_changepoints=10, n_forecasts=30, verbose=True)
    m = NeuralProphet(
        verbose=verbose,
        n_forecasts=1,
        # n_changepoints=5,
        # trend_smoothness=0,
        yearly_seasonality=16,
        # weekly_seasonality=4,
        daily_seasonality=False,
        seasonality_mode='additive',
        # seasonality_mode='multiplicative',
        seasonality_reg=10,
        # learning_rate=1,
        # normalize_y=True,
    )
    # m.set_forecast_in_focus(m.n_forecasts)
    m.fit(df)
    forecast = m.predict(future_periods=365)
    print(sum(abs(m.model.season_params["yearly"].data.numpy())))
    print(sum(abs(m.model.season_params["weekly"].data.numpy())))
    print(m.model.season_params.items())
    if verbose:
        m.plot_components(forecast)
        m.plot(forecast)
        # m.plot(forecast, crop_last_n=365+m.n_forecasts)
        plt.show()



if __name__ == '__main__':
    """
    just used for debugging purposes. 
    should implement proper tests at some point in the future.
    (some test methods might already be deprecated)
    """
    test_eval()
    # test_predict()
    # test_trend()
    # test_ar_net()
    # test_seasons()
