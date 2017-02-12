import os

from parser import *

script_name = os.path.basename(__file__)
config_file = script_name.split('.')[0]+'.cfg'
main(load_config_file(config_file))
