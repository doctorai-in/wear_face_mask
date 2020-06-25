""" Interface to config files and logging. """

CONF_PATH = '/home/ankit/wear_mask_to_face/configs/internal.yaml'

import yaml
import logging
import sys

# Internal config info
with open(CONF_PATH,'r') as infile:
    config = yaml.load(infile.read(), Loader=yaml.SafeLoader)

try:
    logKey = { 'dev' : "INFO" , 'prod' : "ERROR" }
    _level = logKey['dev']
except KeyError:
    _level = "INFO"

logging.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=_level,
                    stream=sys.stdout)
