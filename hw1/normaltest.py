import os

from stats_handler import *

script_name = os.path.basename(__file__)
config_file = script_name.split('.')[0]+'.cfg'
configs =load_normal_config_file(config_file)
normal(configs)
