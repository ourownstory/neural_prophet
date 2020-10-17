import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('logs.log', 'w+')
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
c_format = logging.Formatter('%(levelname)s - %(name)s - %(funcName)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s  - %(funcName)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
log.addHandler(c_handler)
log.addHandler(f_handler)


def set_global_log_level(log_level='NOTSET'):
    if log_level is None or log_level not in ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
        raise ValueError(
            "Please specify a valid log level from: "
            "'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'"
        )
    else:
        log.setLevel(log_level)
        for h in log.handlers:
            h.setLevel(log_level)


from neural_prophet import NeuralProphet
