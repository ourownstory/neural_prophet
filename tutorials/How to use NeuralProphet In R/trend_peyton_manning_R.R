#Fitting a changing trend
#We will use the time series of the log daily page views for the Wikipedia page for Peyton Manning as an example to illustrate how to fit a changing trend.

#First, we load the data:
#If you have not installed reticulate library. 
#Try: install.packages("reticulate")
library(reticulate)
library(ggplot2)
#Installing neural prophet
py_install("NeuralProphet")
conda_create("r-reticulate")
conda_install("r-reticulate", "NeuralProphet")
NeuralProphet <- import("neuralprophet")
df <- read.csv("https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/wp_log_peyton_manning.csv")
#Now we can fit an initial model without any customizations.

#We specify the data frequency to be daily. The model will remember this later when we predict into the future.
m=NeuralProphet$NeuralProphet()
metrics=m$fit(df,freq="D")
#The returned metrics dataframe contains recoded metrics for each training epoch.

#Next, we create a dataframe to predict on. Here, we specify that we want to predict one year into the future and that we want to include the entire history.
future = m$make_future_dataframe(df, periods=as.integer(365), n_historic_predictions=TRUE)
forecast = m$predict(future)

ggplot(data=forecast,mapping=aes(x=forecast$ds,y=forecast$y))+geom_point()+
  geom_line(data = forecast, aes(x = forecast$ds, y = forecast$yhat1), color = "blue") 
#Adjusting Trend
#The default values work fairly well in this example. However, the default of 5 changepoints may not be adequate if the actual change in trend happens to fall in a region between the points.

#Increasing Trend Flexibility
#We can address this by increasing the number of changepoints, giving the trend more flexibility, at the danger of overfitting.

#Let's try what happens if we increase the number of changepoints to 30. Additionally, we can increase the range of data on which we fit trend changepoints to only exlude the last 10 percent (default is 20 percent).

#Further, we can limit the trend flexibility by regularizing it.
m = NeuralProphet$NeuralProphet(
  n_changepoints=30,
  changepoints_range=0.90,    
  trend_reg=1,
)
metrics = m$fit(df, freq="D")
forecast = m$predict(df)
#Looking at the trend rate changes it becomes evident that the trend is overfitting to short-term fluctuations.

#Manual Trend Changepoints
#You can also manually specify the trend changepoints.

#Note: A changepoint will always be added at the beginning. You can ignore it.
m = NeuralProphet$NeuralProphet()
metrics = m$fit(df, freq="D")
forecast = m$predict(df)


