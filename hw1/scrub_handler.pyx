from mpi4py import MPI
from data_handler import *

def scrub(configs):
    logger               = get_logger()

    comm                 = MPI.COMM_WORLD
    rank                 = comm.Get_rank()
    num_process          = comm.Get_size()

    data_file            = configs['data_file']
    noise_file           = configs['noise_file']
    size_notation        = configs['size_notation']
    verbose              = configs['verbose']
    ticks_per_iteration  = configs['ticks_per_iteration']
    memory_per_iteration = configs['memory_per_iteration']
    ret_threshold        = configs['ret_threshold']
    time_drift_threshold = configs['time_drift_threshold']
    size_mult            = configs['size_mult']

    size_per_process     = int(float(memory_per_iteration)/num_process)
    buffer_per_process   = bytearray(size_per_process)

    if rank == 0:
        logger.info('Configs|memoryPerIteration=%i M|ticksPeriteration=%i|sizeNotation=%s|inputFile=%s|verbose=%i'
                    %(memory_per_iteration>>size_mult, ticks_per_iteration, size_notation, data_file, verbose))

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    comm.Barrier()
    file_      = MPI.File.Open(comm, data_file, MPI.MODE_RDONLY)
    noise_file = MPI.File.Open(comm, noise_file, MPI.MODE_WRONLY|MPI.MODE_CREATE)
    size       = file_.Get_size()

    load_iterations   = int(float(size) / memory_per_iteration)

    if rank == 0:
        logger.info('Initializing|fileSize=%i %s|totalLoadIterations=%i|numberOfProcesses=%i|bufferSize=%i %s'
                    %(size>>size_mult, size_notation, load_iterations, num_process, len(buffer_per_process)>>size_mult, size_notation))

    total_time        = 0
    load_file_time    = 0
    process_file_time = 0
    write_file_time   = 0
    bad_data_records  = ''

    total_start_time  = MPI.Wtime()

    for i in range(load_iterations):

        file_offset             = i * memory_per_iteration
        offset_per_process      = rank * size_per_process + file_offset

        load_file_start_time    = MPI.Wtime()
        raw_data                = read_data(file_, offset_per_process, buffer_per_process)
        data                    = format_data(raw_data)
        load_file_time         += MPI.Wtime() - load_file_start_time
        if rank == 0:
            logger.info('Processing file|iteration=%i|totalIterations=%i|numLines=%i' % (i, load_iterations, len(data)))

        process_file_start_time = MPI.Wtime()
        bad_data_records       += scrub_data(rank, data, ticks_per_iteration, ret_threshold)
        process_file_time      += MPI.Wtime() - process_file_start_time


    write_file_start_time = MPI.Wtime()
    if rank == 0:
        logger.info("Writing data start")
    write_data(noise_file, bad_data_records)
    if rank == 0:
        logger.info("Writing data done")



    total_time           += MPI.Wtime() - total_start_time
    write_file_time      += MPI.Wtime() - write_file_start_time

    total_time            = np.array([total_time])
    load_file_time        = np.array([load_file_time])
    process_file_time     = np.array([process_file_time])
    write_file_time       = np.array([write_file_time])
    num_bad_ticks         = np.array([float(len(bad_data_records.split('\n')))])

    max_execution_time    = np.zeros(1)
    min_execution_time    = np.zeros(1)
    max_write_file_time   = np.zeros(1)
    min_write_file_time   = np.zeros(1)
    max_process_file_time = np.zeros(1)
    min_process_file_time = np.zeros(1)
    max_load_file_time    = np.zeros(1)
    min_load_file_time    = np.zeros(1)
    sum_num_bad_ticks     = np.zeros(1)

    comm.Barrier()
    file_.Close()
    noise_file.Close()

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
        logger.info("Timer: total        max - %f, min - %f" % (max_execution_time, min_execution_time))
        logger.info("Timer: load file    max - %f, min - %f" % (max_load_file_time, min_load_file_time))
        logger.info("Timer: process file max - %f, min - %f" % (max_process_file_time, min_process_file_time))
        logger.info("Timer: write file   max - %f, min - %f" % (max_write_file_time, min_write_file_time))
        logger.info("Counter: bad ticks  sum - %i" % (sum_num_bad_ticks))


