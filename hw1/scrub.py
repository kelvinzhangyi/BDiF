import os

from scrub_handler import *

script_name = os.path.basename(__file__)
config_file = script_name.split('.')[0]+'.cfg'
configs =load_scrub_config_file(config_file)
scrub(configs)
