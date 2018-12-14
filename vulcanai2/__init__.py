import logging, logging.config
from os import path
from datetime import datetime

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.ini')
# will create a file wherever you're calling
log_output_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_logfile.log"

logging.config.fileConfig(log_file_path,
                          defaults={'logfilename': log_output_path})
logger = logging.getLogger(__name__)
