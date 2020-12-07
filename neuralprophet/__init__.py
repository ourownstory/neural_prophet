import logging

log = logging.getLogger("nprophet")
log.setLevel("INFO")
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("logs.log", "w+")
# c_handler.setLevel("WARNING")
# f_handler.setLevel("INFO")
# Create formatters and add it to handlers
c_format = logging.Formatter("%(levelname)s: %(name)s - %(funcName)s: %(message)s")
f_format = logging.Formatter("%(asctime)s; %(levelname)s; %(name)s; %(funcName)s; %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
log.addHandler(c_handler)
log.addHandler(f_handler)


from .forecaster import NeuralProphet
from .utils import set_random_seed


def set_log_level(log_level="INFO", include_handlers=False):
    """Set the log level of all underlying logger objects

    Args:
        log_level (str): The log level of the logger objects used for printing procedure status
            updates for debugging/monitoring. Should be one of 'NOTSET', 'DEBUG', 'INFO', 'WARNING',
            'ERROR' or 'CRITICAL'
        include_handlers (bool): include any specified file/stream handlers
    """
    utils.set_logger_level(log, log_level, include_handlers)
