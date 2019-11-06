import logging.config
import os
import torch
import random
import numpy as np


DEFAULT_RANDOM_SEED = 42

log_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'logging.conf')

if not os.path.isfile(log_config_path):
    raise IOError

log_output_path = "logfile.log"

logging.config.fileConfig(log_config_path,
                          disable_existing_loggers=False,
                          defaults={'logfilename': log_output_path})

logger = logging.getLogger(__name__)


def set_global_seed(seed_value):
    """
    Sets all the random seeds, including for torch, GPU, numpy and python.

    Parameters:
        seed_value : int
            The random seed value.

    """
    random.seed(seed_value)
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


set_global_seed(DEFAULT_RANDOM_SEED)

