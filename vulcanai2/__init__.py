import logging, logging.config
from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.ini')
log_output_path = "logfile.log"  # will create a file wherever you're calling

logging.config.fileConfig(log_file_path,
                          defaults={'logfilename': log_output_path})
logger = logging.getLogger(__name__)
