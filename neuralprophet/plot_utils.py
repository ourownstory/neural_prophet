import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch

from neuralprophet import time_dataset

log = logging.getLogger("NP.plotting")


def log_warning_deprecation_plotly(plotting_backend):
    if plotting_backend == "matplotlib":
        log.warning(
            "DeprecationWarning: default plotting_backend will be changed to plotly in a future version. "
            "Switch to plotly by calling `m.set_plotting_backend('plotly')`."
        )


def set_y_as_percent(ax):
    """Set y axis as percentage

    Parameters
    ----------
        ax : matplotlib axis
            Respective y axis element

    Returns
    -------
        matplotlib axis
            Manipulated axis element
    """
    warnings.filterwarnings(
        action="ignore", category=UserWarning
    )  # workaround until there is clear direction how to handle this recent matplotlib bug
    yticks = 100 * ax.get_yticks()
    yticklabels = [f"{y:.4g}%" for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


def predict_one_season(m, quantile, name, n_steps=100, df_name="__df__"):
    """
     Predicts the seasonal component given a number of time steps.

    Parameters
    ----------
         m : NeuralProphet
             Fitted NeuralProphet model
         quantile: float
             The quantile for which the season is predicted
         name: str
             Name of seasonality component
         n_steps: int
             number of prediction steps related to the season frequency
         df_name: str
                Name of dataframe to refer to data params from original keys of train dataframes

        Returns
        -------
            t_i: np.array
                 time scale of predicted seasonal component
            predicted: OrderedDict
                 predicted seasonal component

    """
    config = m.config_season.periods[name]
    t_i = np.arange(n_steps + 1) / float(n_steps)
    features = time_dataset.fourier_series_t(
        t=t_i * config.period, period=config.period, series_order=config.resolution
    )
    features = torch.from_numpy(np.expand_dims(features, 1))

    if df_name == "__df__":
        meta_name_tensor = None
    else:
        meta = OrderedDict()
        meta["df_name"] = [df_name for _ in range(n_steps + 1)]
        meta_name_tensor = torch.tensor([m.model.id_dict[i] for i in meta["df_name"]])

    quantile_index = m.model.quantiles.index(quantile)
    predicted = m.model.seasonality(features=features, name=name, meta=meta_name_tensor)[:, :, quantile_index]
    predicted = predicted.squeeze().detach().numpy()
    if m.config_season.mode == "additive":
        data_params = m.config_normalization.get_data_params(df_name)
        scale = data_params["y"].scale
        predicted = predicted * scale
    return t_i, predicted


def predict_season_from_dates(m, dates, name, quantile, df_name="__df__"):
    """
     Predicts the seasonal component given a date range.

     Parameters
     ----------
         m : NeuralProphet
             Fitted NeuralProphet model
         dates: pd.datetime
             date range for prediction
         name: str
             Name of seasonality component
         quantile: float
             The quantile for which the season is predicted
         df_name: str
             Name of dataframe to refer to data params from original keys of train dataframes

    Returns
    -------
        predicted: OrderedDict
             presdicted seasonal component
    """
    config = m.config_season.periods[name]
    features = time_dataset.fourier_series(dates=dates, period=config.period, series_order=config.resolution)
    features = torch.from_numpy(np.expand_dims(features, 1))
    if m.id_list.__len__() > 1:
        df_name = m.id_list[0]
    if df_name == "__df__":
        meta_name_tensor = None
    else:
        meta = OrderedDict()
        meta["df_name"] = [df_name for _ in range(len(dates))]
        meta_name_tensor = torch.tensor([m.model.id_dict[i] for i in meta["df_name"]])

    quantile_index = m.model.quantiles.index(quantile)
    predicted = m.model.seasonality(features=features, name=name, meta=meta_name_tensor)[:, :, quantile_index]

    predicted = predicted.squeeze().detach().numpy()
    if m.config_season.mode == "additive":
        data_params = m.config_normalization.get_data_params(df_name)
        scale = data_params["y"].scale
        predicted = predicted * scale
    predicted = {name: predicted}
    return predicted


def check_if_configured(m, components, error_flag=False):  # move to utils
    """Check if components were set in the model configuration by the user.

    Parameters
    ----------
        m : NeuralProphet
            Fitted NeuralProphet model
        components : str or list, optional
            name or list of names of components to check

            Options
            ----
            * ``trend``
            * ``trend_rate_change``
            * ``seasonality``
            * ``autoregression``
            * ``lagged_regressors```
            * ``events``
            * ``future_regressors`
            * ``uncertainty``
        error_flag : bool
            Activate to raise a ValueError if component has not been configured

    Returns
    -------
        components
            list of components only including the components set in the model configuration
    """
    invalid_components = []
    if "trend_rate_change" in components and m.model.config_trend.changepoints is None:
        components.remove("trend_rate_change")
        invalid_components.append("trend_rate_change")
    if "seasonality" in components and m.config_season is None:
        components.remove("seasonality")
        invalid_components.append("seasonality")
    if "autoregression" in components and not m.config_ar.n_lags > 0:
        components.remove("autoregression")
        invalid_components.append("autoregression")
    if "lagged_regressors" in components and m.config_lagged_regressors is None:
        components.remove("lagged_regressors")
        invalid_components.append("lagged_regressors")
    if "events" in components and (m.config_events and m.config_country_holidays) is None:
        components.remove("events")
        invalid_components.append("events")
    if "future_regressors" in components and m.config_regressors is None:
        components.remove("future_regressors")
        invalid_components.append("future_regressors")
    if "uncertainty" in components and not len(m.model.quantiles) > 1:
        components.remove("uncertainty")
        invalid_components.append("uncertainty")
    if error_flag and len(invalid_components) != 0:
        raise ValueError(
            f" Selected component(s) {(invalid_components)} for plotting not specified in the model configuration."
        )
    return components


def get_valid_configuration(  # move to utils
    m, components=None, df_name=None, valid_set=None, validator=None, forecast_in_focus=None, quantile=0.5
):
    """Validate and adapt the selected components to be plotted.

    Parameters
    ----------
        m : NeuralProphet
            Fitted NeuralProphet model
        components : str or list, optional
            name or list of names of components to validate and adapt
        df_name: str
            ID from time series that should be plotted
        valid_set : str or list, optional
            name or list of names of components that are defined as valid option

            Options
            ----
             * (default)``None``:  All components the user set in the model configuration are validated and adapted
            * ``trend``
            * ``seasonality``
            * ``autoregression``
            * ``lagged_regressors``
            * ``future_regressors``
            * ``events``
            * ``uncertainty``
        validator: str
            specifies the validation purpose to customize

            Options
            ----
            * ``plot_parameters``: customize for plot_parameters() function
            * ``plot_components``: customize for plot_components() function
        forecast_in_focus: int
            optinal, i-th step ahead forecast to plot

            Note
            ----
            None (default): plot self.highlight_forecast_step_n by default
        quantile: float
            The quantile for which the model parameters are to be plotted

            Note
            ----
            0.5 (default):  Parameters will be plotted for the median quantile.

    Returns
    -------
        valid_configuration: dict
            dict of validated components and values to be plotted
    """
    if type(valid_set) is not list:
        valid_set = [valid_set]

    if components is None:
        components = valid_set
        components = check_if_configured(m=m, components=components)
    else:
        if type(components) is not list:
            components = [components]
        components = [comp.lower() for comp in components]
        for comp in components:
            if comp not in valid_set:
                raise ValueError(
                    f" Selected component {comp} is either mis-spelled or not an available "
                    f"option for this function."
                )
        components = check_if_configured(m=m, components=components, error_flag=True)
    if validator is None:
        raise ValueError("Specify a validator from the available options")
    # Adapt Normalization
    if validator == "plot_parameters":
        # Set to True in case of local normalization and unknown_data_params is not True
        overwriting_unknown_data_normalization = False
        if m.config_normalization.global_normalization:
            if df_name is None and m.id_list.__len__() == 1:
                df_name = "__df__"
            elif df_name is None and m.id_list.__len__() > 1:
                df_name = m.id_list[0]
            else:
                log.debug("Global normalization set - ignoring given df_name for normalization")
        else:
            if df_name is None:
                if m.id_list.__len__() > 1:
                    if (
                        m.model.config_season.global_local == "local"
                        or m.model.config_trend.trend_global_local == "local"
                    ):
                        df_name = m.id_list
                        log.warning(
                            "Glocal model set with > 1 time series in the pd.DataFrame. Plotting components of mean time series and quants. "
                        )
                    else:
                        df_name = m.id_list[0]
                        log.warning(
                            "Local model set with > 1 time series in the pd.DataFrame. Plotting components of first time series. "
                        )
                else:
                    log.warning("Local normalization set, but df_name is None. Using global data params instead.")
                    df_name = "__df__"
                if not m.config_normalization.unknown_data_normalization:
                    m.config_normalization.unknown_data_normalization = True
                    overwriting_unknown_data_normalization = True
            elif df_name not in m.config_normalization.local_data_params:
                log.warning(
                    f"Local normalization set, but df_name '{df_name}' not found. Using global data params instead."
                )
                df_name = "__df__"
                if not m.config_normalization.unknown_data_normalization:
                    m.config_normalization.unknown_data_normalization = True
                    overwriting_unknown_data_normalization = True
            else:
                log.debug(f"Local normalization set. Data params for {df_name} will be used to denormalize.")

    # Identify components to be plotted
    # as dict, minimum: {plot_name}
    plot_components = []
    if validator == "plot_parameters":
        quantile_index = m.model.quantiles.index(quantile)

    # Plot trend
    if "trend" in components:
        plot_components.append({"plot_name": "Trend", "comp_name": "trend"})
    if "trend_rate_change" in components:
        plot_components.append({"plot_name": "Trend Rate Change"})

    # Plot  seasonalities, if present
    if "seasonality" in components:
        for name in m.config_season.periods:
            if validator == "plot_components":
                plot_components.append(
                    {
                        "plot_name": f"{name} seasonality",
                        "comp_name": name,
                    }
                )
            elif validator == "plot_parameters":
                plot_components.append({"plot_name": "seasonality", "comp_name": name})

    # AR
    if "autoregression" in components:
        if validator == "plot_components":
            if forecast_in_focus is None:
                plot_components.append(
                    {
                        "plot_name": "Auto-Regression",
                        "comp_name": "ar",
                        "num_overplot": m.n_forecasts,
                        "bar": True,
                    }
                )
            else:
                plot_components.append(
                    {
                        "plot_name": f"AR ({forecast_in_focus})-ahead",
                        "comp_name": f"ar{forecast_in_focus}",
                    }
                )
        elif validator == "plot_parameters":
            plot_components.append(
                {
                    "plot_name": "lagged weights",
                    "comp_name": "AR",
                    "weights": m.model.ar_weights.detach().numpy(),
                    "focus": forecast_in_focus,
                }
            )

    # Add lagged regressors
    lagged_scalar_regressors = []
    if "lagged_regressors" in components:
        if validator == "plot_components":
            if forecast_in_focus is None:
                for name in m.config_lagged_regressors.keys():
                    plot_components.append(
                        {
                            "plot_name": f'Lagged Regressor "{name}"',
                            "comp_name": f"lagged_regressor_{name}",
                            "num_overplot": m.n_forecasts,
                            "bar": True,
                        }
                    )
            else:
                for name in m.config_lagged_regressors.keys():
                    plot_components.append(
                        {
                            "plot_name": f'Lagged Regressor "{name}" ({forecast_in_focus})-ahead',
                            "comp_name": f"lagged_regressor_{name}{forecast_in_focus}",
                        }
                    )
        elif validator == "plot_parameters":
            for name in m.config_lagged_regressors.keys():
                if m.config_lagged_regressors[name].as_scalar:
                    lagged_scalar_regressors.append((name, m.model.get_covar_weights(name).detach().numpy()))
                else:
                    plot_components.append(
                        {
                            "plot_name": "lagged weights",
                            "comp_name": f'Lagged Regressor "{name}"',
                            "weights": m.model.get_covar_weights(name).detach().numpy(),
                            "focus": forecast_in_focus,
                        }
                    )

    # Add Events
    additive_events = []
    multiplicative_events = []
    if "events" in components:
        additive_events_flag = False
        muliplicative_events_flag = False
        for event, configs in m.config_events.items():
            if validator == "plot_components" and configs.mode == "additive":
                additive_events_flag = True
            elif validator == "plot_components" and configs.mode == "multiplicative":
                muliplicative_events_flag = True
            elif validator == "plot_parameters":
                event_params = m.model.get_event_weights(event)
                weight_list = [(key, param.detach().numpy()[quantile_index, :]) for key, param in event_params.items()]
                if configs.mode == "additive":
                    additive_events = additive_events + weight_list
                elif configs.mode == "multiplicative":
                    multiplicative_events = multiplicative_events + weight_list

        for country_holiday in m.config_country_holidays.holiday_names:
            if validator == "plot_components" and m.config_country_holidays.mode == "additive":
                additive_events_flag = True
            elif validator == "plot_components" and m.config_country_holidays.mode == "multiplicative":
                muliplicative_events_flag = True
            elif validator == "plot_parameters":
                event_params = m.model.get_event_weights(country_holiday)
                weight_list = [(key, param.detach().numpy()[quantile_index, :]) for key, param in event_params.items()]
                if m.config_country_holidays.mode == "additive":
                    additive_events = additive_events + weight_list
                elif m.config_country_holidays.mode == "multiplicative":
                    multiplicative_events = multiplicative_events + weight_list

        if additive_events_flag:
            plot_components.append(
                {
                    "plot_name": "Additive Events",
                    "comp_name": "events_additive",
                }
            )
        if muliplicative_events_flag:
            plot_components.append(
                {
                    "plot_name": "Multiplicative Events",
                    "comp_name": "events_multiplicative",
                    "multiplicative": True,
                }
            )

    # Add Regressors
    additive_future_regressors = []
    multiplicative_future_regressors = []
    if "future_regressors" in components:
        for regressor, configs in m.config_regressors.items():
            if validator == "plot_components" and configs.mode == "additive":
                plot_components.append(
                    {
                        "plot_name": "Additive Future Regressors",
                        "comp_name": "future_regressors_additive",
                    }
                )
            elif validator == "plot_components" and configs.mode == "multiplicative":
                plot_components.append(
                    {
                        "plot_name": "Multiplicative Future Regressors",
                        "comp_name": "future_regressors_multiplicative",
                        "multiplicative": True,
                    }
                )
            elif validator == "plot_parameters":
                regressor_param = m.model.get_reg_weights(regressor)[quantile_index, :]
                if configs.mode == "additive":
                    additive_future_regressors.append((regressor, regressor_param.detach().numpy()))
                elif configs.mode == "multiplicative":
                    multiplicative_future_regressors.append((regressor, regressor_param.detach().numpy()))

    # Plot  quantiles as a separate component, if present
    # If multiple steps in the future are predicted, only plot quantiles if highlight_forecast_step_n is set
    if (
        "quantiles" in components
        and validator == "plot_components"
        and len(m.model.quantiles) > 1
        and forecast_in_focus is None
    ):
        if len(m.config_train.quantiles) > 1 and (
            m.n_forecasts > 1 or m.config_ar.n_lags > 0
        ):  # rather query if n_forecasts >1 than n_lags>1
            raise ValueError(
                "Please specify step_number using the highlight_nth_step_ahead_of_each_forecast function"
                " for quantiles plotting when autoregression enabled."
            )
        for i in range(1, len(m.model.quantiles)):
            plot_components.append(
                {
                    "plot_name": "Uncertainty",
                    "comp_name": f"yhat1 {round(m.model.quantiles[i] * 100, 1)}%",
                    "fill": True,
                }
            )
    elif (
        "uncertainty" in components
        and validator == "plot_components"
        and len(m.model.quantiles) > 1
        and forecast_in_focus is not None
    ):
        for i in range(1, len(m.model.quantiles)):
            plot_components.append(
                {
                    "plot_name": "Uncertainty",
                    "comp_name": f"yhat{forecast_in_focus} {round(m.model.quantiles[i] * 100, 1)}%",
                    "fill": True,
                }
            )
    if validator == "plot_parameters":
        if len(additive_future_regressors) > 0:
            plot_components.append({"plot_name": "Additive future regressor"})
        if len(multiplicative_future_regressors) > 0:
            plot_components.append({"plot_name": "Multiplicative future regressor"})
        if len(lagged_scalar_regressors) > 0:
            plot_components.append({"plot_name": "Lagged scalar regressor"})
        if len(additive_events) > 0:
            data_params = m.config_normalization.get_data_params(df_name)
            scale = data_params["y"].scale
            additive_events = [(key, weight * scale) for (key, weight) in additive_events]
            plot_components.append({"plot_name": "Additive event"})
        if len(multiplicative_events) > 0:
            plot_components.append({"plot_name": "Multiplicative event"})

        valid_configuration = {
            "components_list": plot_components,
            "additive_future_regressors": additive_future_regressors,
            "additive_events": additive_events,
            "multiplicative_future_regressors": multiplicative_future_regressors,
            "multiplicative_events": multiplicative_events,
            "lagged_scalar_regressors": lagged_scalar_regressors,
            "overwriting_unknown_data_normalization": overwriting_unknown_data_normalization,
            "df_name": df_name,
        }
    elif validator == "plot_components":
        valid_configuration = {
            "components_list": plot_components,
        }
    return valid_configuration
