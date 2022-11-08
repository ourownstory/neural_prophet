import logging

# matplotlib is known to spam logs, thus we set the logging level to warning
logging.getLogger("matplotlib").setLevel(logging.WARNING)
