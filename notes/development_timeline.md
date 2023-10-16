Development Timeline for NeuralProphet
v0.1 Alpha [Initial Release - 2020]

NeuralProphet's journey began with its alpha release, which marked its initial presence in the time-series forecasting landscape. While functional, this version lacked some essential features and may have contained bugs.
v0.2 to v0.5 Beta NeuralProphet [Current Stage]

Modelling Capabilities

Trend modeling with piecewise linear changepoints.
Auto-regression for univariate and multi-step ahead forecasts.
Seasonalities based on Fourier terms.
Optional hidden layers for auto-regression.
Basic missing data management with automatic imputation.
Preliminary automatic hyperparameter selection.
Customizable metrics for forecasting accuracy.
Training with evaluation on a dedicated holdout set.
Integration of events and holidays.
Support for exogenous variables as covariate inputs.
Simple uncertainty estimation.
User Interface

The current version of NeuralProphet offers a straightforward package with limited capabilities, aiming for ease of use.
It shares similarities with the basic features of Facebook Prophet.
Accompanying Products

Quickstart documentation and examples to help users get started.
Benchmarking tools for assessing accuracy and execution time.
A small collection of sample time-series datasets for testing and demonstration purposes.
v1.0 NeuralProphet

Modelling Capabilities

Enhanced automatic hyperparameter selection for more intelligent forecasting.
Diverse options for trend modeling, data normalization, and seasonality computation (e.g., rolling, local seasonality).
Integration of traditional models like ETS and SARIMA.
Component-wise uncertainty estimation to provide a clearer picture of prediction quality.
User Interface

A more user-friendly interface with increased control over settings, including trend changepoint times.
Rich analytics and advanced plotting features.
Feedback mechanisms guiding users on how to optimize hyperparameters.
Seamless integration with Time-Series Preprocessing tools.
Accompanying Products

A comprehensive repository of diverse time-series datasets for research and application.
Professional-grade documentation, including an extensive set of tutorials for users at all skill levels.
v2.0 Redesigned - Modular Framework

Modelling Capabilities

A complete overhaul of the code structure to create a modular framework.
Introduction of more powerful modeling options, including recurrent, convolutional, and attention-based models.
User Interface

Advanced tools for understanding the model and its input-output mapping.
Integration with relevant interfaces and libraries such as PyTorch metrics, Tensorboard, scikit-learn, and more.
Accompanying Products

Availability of pre-trained models for expedited forecasting tasks.
v3.0 User-Friendly Web Interface

Changes

Introducing an alternative visual web-based interface with the possibility of cloud-based execution.
