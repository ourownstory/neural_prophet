import logging

log = logging.getLogger("nprophet")
log.setLevel("INFO")
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('logs.log', 'w+')
# c_handler.setLevel("WARNING")
# f_handler.setLevel("INFO")
# Create formatters and add it to handlers
c_format = logging.Formatter('%(levelname)s - %(name)s - %(funcName)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s  - %(funcName)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
log.addHandler(c_handler)
log.addHandler(f_handler)


def set_global_log_level(log_level=None):
    if log_level is None:
        log.warning("Failed to set global log_level to None.")
    elif log_level not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
                           10, 20, 30, 40, 50):
        log.error(
            "Failed to set global log_level to {}."
            "Please specify a valid log level from: "
            "'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'"
            "".format(log_level)
        )
    else:
        log.setLevel(log_level)
        # for h in log.handlers:
        #     h.setLevel(log_level)
        log.debug("Set log level to {}".format(log_level))


from neural_prophet import NeuralProphet
