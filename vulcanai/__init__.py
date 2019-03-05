import logging.config
import os

log_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'logging.conf')

if not os.path.isfile(log_config_path):
    raise IOError

log_output_path = "logfile.log"

logging.config.fileConfig(log_config_path,
                          disable_existing_loggers=False,
                          defaults={'logfilename': log_output_path})

logger = logging.getLogger(__name__)
