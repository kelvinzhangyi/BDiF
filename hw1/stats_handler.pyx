from data_handler import *

from mpi4py import MPI

def normal(configs):
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
    size       = file_.Get_size()

    load_iterations   = int(float(size) / memory_per_iteration)

    noise         = pd.read_csv(noise_file, header=None)
    noise.columns = ['index', 'timestamp', 'price', 'volume', 'explanation']
    p_values      = []

    for i in range(load_iterations):

        file_offset           = i * memory_per_iteration
        offset_per_process    = rank * size_per_process + file_offset

        raw_data              = read_data(file_, offset_per_process, buffer_per_process)
        data                  = format_data(raw_data)

        if rank == 0:
            logger.info('Processing file|iteration=%i|totalIterations=%i|numLines=%i' % (i, load_iterations, len(data)))

        p_values.extend(normaltest(rank, data, noise, ticks_per_iteration))

    file_.Close()

    p_values      = np.array(p_values)
    p_values      = p_values[~np.isnan(p_values)]
    p_values      = p_values[p_values<=1]
    max_pvalue    = np.array(p_values.max())
    min_pvalue    = np.array(p_values.min())
    sum_pvalue    = np.array(p_values.sum())
    normal        = np.array([float(len(p_values[p_values>0.05]))])
    total         = np.array([float(len(p_values))])

    normal_global = np.zeros(1)
    total_global  = np.zeros(1)

    max_pvalue_global = np.zeros(1)
    min_pvalue_global = np.zeros(1)
    sum_pvalue_global = np.zeros(1)

    comm.Reduce([max_pvalue, MPI.DOUBLE], [max_pvalue_global, MPI.DOUBLE], op=MPI.MAX, root=0)
    comm.Reduce([min_pvalue, MPI.DOUBLE], [min_pvalue_global, MPI.DOUBLE], op=MPI.MIN, root=0)
    comm.Reduce([sum_pvalue, MPI.DOUBLE], [sum_pvalue_global, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([normal, MPI.DOUBLE], [normal_global, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([total,  MPI.DOUBLE], [total_global, MPI.DOUBLE],  op=MPI.SUM, root=0)

    if rank == 0:
        logger.info("Normal check: mean p-value: %.2f, max p_value: %.2f, min p_value: %.2f" % (sum_pvalue_global/total_global, max_pvalue_global, min_pvalue_global))
        logger.info("Normal check: %.2f out of %.2f total sampled are normal." % (normal_global, total_global))


