import logging
import warnings

log = logging.getLogger("NP.plotting")


def log_warning_deprecation_plotly(plotting_backend):
    if plotting_backend == "matplotlib":
        log.warning(
            "DeprecationWarning: default plotting_backend will be changed to plotly in a future version. "
            "Switch to plotly by calling `m.set_plotting_backend('plotly')`."
        )
