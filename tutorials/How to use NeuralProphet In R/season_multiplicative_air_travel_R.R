#Multiplicative Seasonality in R
#By default NeuralProphet fits additive seasonalities, 
#meaning the effect of the seasonality is added to the trend to get
#the forecast. This time series of the number of air passengers 
#is an example of when additive seasonality does not work:

#If you have not installed reticulate library. 
#Try: install.packages("reticulate")
library(reticulate)
library(ggplot2)
#Installing neuralprophet
py_install("NeuralProphet")
NeuralProphet <- import("neuralprophet")
df <- read.csv("https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/air_passengers.csv")
m=NeuralProphet$NeuralProphet()
metrics=m$fit(df,freq="MS")
#Make sure to change the parameters to there type when calling the library function
future=m$make_future_dataframe(df,periods=as.integer(50),n_historic_predictions=as.integer(144))
forecast = m$predict(future)

#Using ggplot 2 to plot
ggplot(data=forecast,mapping=aes(x=forecast$ds,y=forecast$y))+geom_point()+
  geom_line(data = forecast, aes(x = forecast$ds, y = forecast$yhat1), color = "blue") 
#This time series has a clear yearly cycle, but the seasonality in the 
#forecast is too large at the start of the time series and too small at 
#the end. In this time series, the seasonality is not a constant additive 
#factor as assumed by NeuralProphet, rather it grows with the trend. This 
#is multiplicative seasonality.

#NeuralProphet can model multiplicative seasonality by setting 
#seasonality_mode="multiplicative" in the input arguments:
m = NeuralProphet$NeuralProphet(seasonality_mode="multiplicative")
metrics = m$fit(df, freq="MS")

future=m$make_future_dataframe(df,periods=as.integer(50),n_historic_predictions=as.integer(144))
forecast = m$predict(future)
ggplot(data=forecast,mapping=aes(x=forecast$ds,y=forecast$y))+geom_point()+
  geom_line(data = forecast, aes(x = forecast$ds, y = forecast$yhat1), color = "blue") 


ggplot(data=forecast,mapping=aes(x=forecast$ds,y=forecast$trend))+geom_line(color="blue")
#day of year needs manipulation

#Note that the seasonality is only fit on data occuring at the start 
#of the month. Thus, the plotted values for seasonality inbetween months
#may take on random values.

#Setting seasonality_mode="multiplicative" will model all seasonalities 
#as multiplicative, including custom seasonalities added with add_seasonality.
