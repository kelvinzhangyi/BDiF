import logging
import subprocess
import configparser
import numpy as np

from mpi4py import MPI

SIZE_MAP  = {'B':0, 'K':10,'M':20,'G':30}

logger      = logging.getLogger()
handler     = logging.StreamHandler()
formatter   = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def date_converter(data):
    try:
        time = data[1].split(':')[1:4]
        time = float(time[0])*3600 + float(time[1])*60 + float(time[2])
        return time

    except:
        return 0

def reset_file_flag(input_file):
    proc = subprocess.run(['sed', '-i', '-e', 's/^./0/g', input_file])
    if proc.returncode == 0:
        logger.info('Reset file flag a success.')
    else:
        logger.info('Reset file flag a failed.')

def read_data(input_file, offset ,buf):

    input_file.Read_at(offset, buf)
    data      = buf.decode().split('\n')

    positions = offset + np.array([len(line)+1 for line in data[:-1]]).cumsum()[:-1]
    positions = positions.reshape(1, len(positions))

    data      = [x.split(',') for x in data[1:-1]]
    data      = [x if len(x) == 4 else ['0', '0', '0', '0'] for x in data]
    data      = np.array(data)
    data[:,1] = np.apply_along_axis(date_converter, 1, data)
    data      = np.hstack((data, positions.T))
    data      = np.core.records.fromarrays(data.transpose(), \
                                           names='flag,timestamp,price,volume,offset', \
                                           formats='i4,f16,f16,i8,i4')
    return data


def parse_data(data, ticks_per_iteration, ret_threshold=0.5, time_drift_threshold=0.5):

    iterations      = int(np.ceil(float(len(data)) / ticks_per_iteration))
    bad_data_offset = np.array([])

    for i in range(iterations):

        if i == (iterations-1):
            data_slice = data[i*ticks_per_iteration:]
        else:
            data_slice = data[i*ticks_per_iteration:(i+1)*ticks_per_iteration]

        # fix missing value issue by inserting a 0 at time to be picked up later
        # convert datetime to timestamps

        data_slice = np.sort(data_slice, order='timestamp')
        ret        = np.divide(data_slice['price'][1:], data_slice['price'][:-1]) - 1
        time_drift = data_slice['timestamp'][1:] - data_slice['timestamp'][:-1]

        f = (data_slice['price'] <= 0)  | (data_slice['volume'] <= 0) | (data_slice['timestamp'] == 0)
        g = (ret > ret_threshold) | (ret < (-1) *ret_threshold)
        h = (time_drift > time_drift_threshold) | (time_drift == 0)

        logger.debug('Len of bad_data_offset: %s' % len(bad_data_offset))
        bad_data_offset = np.append(bad_data_offset, data_slice['offset'][f])
        logger.debug('Len of bad_data_offset|after negative points: %s' % len(bad_data_offset))
        bad_data_offset = np.append(bad_data_offset, data_slice['offset'][1:][g])
        logger.debug('Len of bad_data_offset|after return outlier: %s' % len(bad_data_offset))
        bad_data_offset = np.append(bad_data_offset, data_slice['offset'][1:][h])
        logger.debug('Len of bad_data_offset|after time outlier: %s' % len(bad_data_offset))

    return bad_data_offset

def flag_data(input_file, offsets):
    for i in offsets:
        input_file.Write_at(i, bytearray(b'1'))


def load_config_file(config_file):
    # loading configs
    config = configparser.RawConfigParser()
    config.read(config_file)
    configs = {}

    configs['data_file']            = config.get('default', 'data_file')
    configs['size_notation']        = config.get('default', 'size_notation')

    configs['verbose']              = int(config.get('default', 'verbose'))
    configs['reset_file']           = int(config.get('default', 'reset_file'))
    configs['ticks_per_iteration']  = int(config.get('default', 'ticks_per_iteration'))
    configs['memory_per_iteration'] = int(config.get('default', 'memory_per_iteration'))<<20 # assume the config is in megabyte

    configs['ret_threshold']        = float(config.get('default', 'ret_threshold'))
    configs['time_drift_threshold'] = float(config.get('default', 'time_drift_threshold'))

    configs['size_mult']            = SIZE_MAP[configs['size_notation']]

    return configs

def main(configs):
    comm        = MPI.COMM_WORLD
    rank        = comm.Get_rank()
    num_process = comm.Get_size()

    data_file            = configs['data_file']
    size_notation        = configs['size_notation']
    verbose              = configs['verbose']
    reset_file           = configs['reset_file']
    ticks_per_iteration  = configs['ticks_per_iteration']
    memory_per_iteration = configs['memory_per_iteration']
    ret_threshold        = configs['ret_threshold']
    time_drift_threshold = configs['time_drift_threshold']
    size_mult            = configs['size_mult']

    size_per_process     = int(float(memory_per_iteration)/num_process)
    buffer_per_process   = bytearray(size_per_process)

    if rank == 0:
        if reset_file:
            reset_file_flag(data_file)
        logger.info('Configs|memoryPerIteration=%i M|ticksPeriteration=%i|sizeNotation=%s|inputFile=%s|verbose=%i'
                    %(memory_per_iteration>>size_mult, ticks_per_iteration, size_notation, data_file, verbose))

    if verbose:
        logger.setLevel(logging.DEBUG)

    comm.Barrier()
    file_  = MPI.File.Open(comm, data_file, MPI.MODE_RDWR)
    size   = file_.Get_size()

    load_iterations = int(float(size) / memory_per_iteration)

    total_time        = 0
    load_file_time    = 0
    process_file_time = 0
    write_file_time   = 0
    bad_data_offset   = np.array([])

    if rank == 0:
        logger.info('Initializing|fileSize=%i %s|totalLoadIterations=%i|numberOfProcesses=%i|bufferSize=%i %s'
                    %(size>>size_mult, size_notation, load_iterations, num_process, len(buffer_per_process)>>size_mult, size_notation))

    total_start_time = MPI.Wtime()

    for i in range(load_iterations):

        file_offset             = i * memory_per_iteration
        offset_per_process      = rank*size_per_process + file_offset

        load_file_start_time    = MPI.Wtime()
        data                    = read_data(file_, offset_per_process, buffer_per_process)
        load_file_time         += MPI.Wtime() - load_file_start_time

        if rank == 0:
            logger.info('Processing file|iteration=%i|totalIterations=%i|numLines=%i' % (i, load_iterations, len(data)))

        process_file_start_time = MPI.Wtime()
        bad_data_offset         = np.append(bad_data_offset, parse_data(data, ticks_per_iteration, ret_threshold, time_drift_threshold))
        process_file_time      += MPI.Wtime() - process_file_start_time

    write_file_start_time = MPI.Wtime()
    flag_data(file_, bad_data_offset)
    file_.Close()

    total_time           += MPI.Wtime() - total_start_time
    write_file_time      += MPI.Wtime() - write_file_start_time

    total_time            = np.array([total_time])
    load_file_time        = np.array([load_file_time])
    process_file_time     = np.array([process_file_time])
    write_file_time       = np.array([write_file_time])
    num_bad_ticks         = np.array([float(len(bad_data_offset))])

    max_execution_time    = np.zeros(1)
    min_execution_time    = np.zeros(1)
    max_write_file_time   = np.zeros(1)
    min_write_file_time   = np.zeros(1)
    max_process_file_time = np.zeros(1)
    min_process_file_time = np.zeros(1)
    max_load_file_time    = np.zeros(1)
    min_load_file_time    = np.zeros(1)
    sum_num_bad_ticks     = np.zeros(1)

    comm.Reduce([total_time, MPI.DOUBLE], [max_execution_time, MPI.DOUBLE],   op=MPI.MAX, root=0)
    comm.Reduce([total_time, MPI.DOUBLE], [min_execution_time, MPI.DOUBLE],   op=MPI.MIN, root=0)
    comm.Reduce([process_file_time, MPI.DOUBLE], [max_process_file_time, MPI.DOUBLE],   op=MPI.MAX, root=0)
    comm.Reduce([process_file_time, MPI.DOUBLE], [min_process_file_time, MPI.DOUBLE],   op=MPI.MIN, root=0)
    comm.Reduce([load_file_time, MPI.DOUBLE], [max_load_file_time, MPI.DOUBLE],   op=MPI.MAX, root=0)
    comm.Reduce([load_file_time, MPI.DOUBLE], [min_load_file_time, MPI.DOUBLE],   op=MPI.MIN, root=0)
    comm.Reduce([write_file_time, MPI.DOUBLE], [max_write_file_time, MPI.DOUBLE],   op=MPI.MAX, root=0)
    comm.Reduce([write_file_time, MPI.DOUBLE], [min_write_file_time, MPI.DOUBLE],   op=MPI.MIN, root=0)
    comm.Reduce([num_bad_ticks, MPI.DOUBLE], [sum_num_bad_ticks, MPI.DOUBLE],   op=MPI.SUM, root=0)

    if rank == 0:
        logger.info("Timer: total        max - %.2f, min - %.2f" % (max_execution_time, min_execution_time))
        logger.info("Timer: load file    max - %.2f, min - %.2f" % (max_load_file_time, min_load_file_time))
        logger.info("Timer: process file max - %.2f, min - %.2f" % (max_process_file_time, min_process_file_time))
        logger.info("Timer: write file   max - %.2f, min - %.2f" % (max_write_file_time, min_write_file_time))
        logger.info("Counter: bad ticks  sum - %i" % (sum_num_bad_ticks))

