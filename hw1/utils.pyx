import logging
import datetime
import configparser

SIZE_MAP            = {'B':0, 'K':10,'M':20,'G':30}
BAD_DATA_REASON_MAP = {'1': 'Invalid Price', '2': 'Invalid Volume', '3': 'Invalid Date', '4': 'Outlier', '5': 'Duplicate'}

def get_logger():
    logger      = logging.getLogger()
    handler     = logging.StreamHandler()
    formatter   = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_scrub_config_file(config_file):
    # loading configs
    config = configparser.RawConfigParser()
    config.read(config_file)
    configs = {}

    configs['data_file']            = config.get('default', 'data_file')
    configs['noise_file']           = config.get('default', 'noise_file')
    configs['size_notation']        = config.get('default', 'size_notation')

    configs['verbose']              = int(config.get('default', 'verbose'))
    configs['ticks_per_iteration']  = int(config.get('default', 'ticks_per_iteration'))
    configs['memory_per_iteration'] = int(config.get('default', 'memory_per_iteration'))<<20 # assume the config is in megabyte

    configs['ret_threshold']        = float(config.get('default', 'ret_threshold'))
    configs['time_drift_threshold'] = float(config.get('default', 'time_drift_threshold'))

    configs['size_mult']            = SIZE_MAP[configs['size_notation']]

    return configs

def load_normal_config_file(config_file):
    # loading configs
    config = configparser.RawConfigParser()
    config.read(config_file)
    configs = {}

    configs['data_file']            = config.get('default', 'data_file')
    configs['noise_file']           = config.get('default', 'noise_file')
    configs['size_notation']        = config.get('default', 'size_notation')

    configs['verbose']              = int(config.get('default', 'verbose'))
    configs['ticks_per_iteration']  = int(config.get('default', 'ticks_per_iteration'))
    configs['memory_per_iteration'] = int(config.get('default', 'memory_per_iteration'))<<20 # assume the config is in megabyte

    configs['size_mult']            = SIZE_MAP[configs['size_notation']]

    return configs

def date_converter(data):
    try:
        ts = datetime.datetime.strptime(data[0], '%Y%m%d:%H:%M:%S.%f').timestamp()

        return ts

    except:
        return 0

