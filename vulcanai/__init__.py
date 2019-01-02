import logging, logging.config
from os import path
from datetime import datetime

log_config_path = path.join(path.dirname(path.abspath(__file__)), 'log.ini')
# will create a file wherever you're calling
log_output_path = "logfile.log"

logging.config.fileConfig(log_config_path,
                          defaults={'logfilename': log_output_path})
logger = logging.getLogger(__name__)
