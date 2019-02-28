import logging.config
from os import path

log_config_path = path.join(__name__, 'logging.conf')
log_output_path = "logfile.log"

logging.config.fileConfig(log_config_path,
                          disable_existing_loggers=False,
                          defaults={'logfilename': log_output_path})

logger = logging.getLogger(__name__)
