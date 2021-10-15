.. NeuralProphet documentation master file, created by
   sphinx-quickstart on Tue Oct 12 13:27:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========================================
NeuralProphet
=========================================

*A simple time series forecasting framework based on Neural Networks in PyTorch.*


Documentation
==============

.. toctree::
   :maxdepth: 1


   Quick Start Guide<model/README.md>
   Model Overview<model/model-overview>
   Changes from prophet<model/changes-from-prophet>
   Trend<model/trend>
   Seasonality<model/seasonality>
   Auto-regression<model/auto-regression>
   Lagged-regression<model/lagged-regressors>
   Events<model/events>
   Future-regression<model/future-regressors>
   Hyperparameter-selection<model/hyperparameter-selection>
   Contribution<model/contribute>


Get started with Python Notebooks
==================================

.. toctree::
   :maxdepth: 1
   :caption: Feature Tutorials

   example_links/autoregression_yosemite_temps.nblink
   example_links/crossvalidation.nblink
   example_links/lagged_covariates_energy_ercot.nblink
   example_links/events_holidays_peyton_manning.nblink
   example_links/season_multiplicative_air_travel.nblink
   example_links/sparse_autoregression_yosemite_temps.nblink
   example_links/sub_daily_data_yosemite_temps.nblink
   example_links/trend_peyton_manning.nblink
   

.. toctree::
   :maxdepth: 1
   :caption: Application Tutorials   

   example_links/energy_data_example.nblink


Core Modules
=============

.. toctree::
   :maxdepth: 1
   :caption: Core Modules

   __init__.py <module_links/__init__>
   configure.py <module_links/configure>
   df_utils.py <module_links/df_utils>
   forecaster.py <module_links/forecaster>
   hdays.py <module_links/hdays>
   metrics.py <module_links/metrics>
   plot_forecaster.py <module_links/plot_forecast>
   plot_model_parameters.py <module_links/plot_model_parameters>
   time_dataset.py <module_links/time_dataset>
   time_net.py <module_links/time_net>
   utils_torch.py <module_links/utils_torch>
   utils.py <module_links/utils>

