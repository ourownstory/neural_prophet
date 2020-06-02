import pandas as pd
from fbprophet import Prophet



def run_prophet():
    """
    Purpose of this function is solely to analyze Phrophet's methods by using Pycharms code tools,
    to find definitions and usages of variables and methods called within the code
    """
    df = pd.read_csv('data/example_wp_log_peyton_manning.csv')
    # print(df.head())
    # print(df.tail())

    m = Prophet()
    m = m.fit(df)

    future = m.make_future_dataframe(periods=365)
    # print(future.tail())

    forecast = m.predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    fig1 = m.plot(forecast)
    fig1.show()
    fig2 = m.plot_components(forecast)


if __name__ == '__main__':
    run_prophet()
