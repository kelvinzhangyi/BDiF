import numpy as np
import pandas as pd
import scipy.stats as stats

from utils import *

def read_data(input_file, offset ,buf):
    input_file.Read_at(offset, buf)
    data = buf.decode().split('\n')[1:-1]

    return data

def format_data(raw_data):

    data  = [x.split(',') for x in raw_data]
    data  = [x if len(x) == 3 else ['0', '0', '0'] for x in data]
    data  = np.array(data)

    time  = np.apply_along_axis(date_converter, 1, data)
    time  = time.reshape(1, len(time))
    data  = np.hstack((data, time.T))

    index = np.arange(len(data))
    index = index.reshape(1, len(index))
    data  = np.hstack((data, index.T))

    flag  = np.zeros(len(data), dtype='i4')
    flag  = flag.reshape(1, len(flag))
    data  = np.hstack((data, flag.T))
    data  = np.core.records.fromarrays(data.transpose(), \
                                       names='timestamp,price,volume,time,index,flag', \
                                       formats='U24,f8,i8,f8,i8,i4')
    return data

def normaltest(rank, data, noise, ticks_per_iteration):

    noise_timestamps = noise['timestamp'].tolist()
    good_timestamps  = [timestamp for timestamp in data['timestamp'] if timestamp not in noise_timestamps]

    data       = data[np.in1d(data['timestamp'], good_timestamps)]
    iterations = int(np.ceil(float(len(data)) / ticks_per_iteration))
    p_values   = []

    for i in range(iterations):

        if i == (iterations-1):
            data_slice = data[i*ticks_per_iteration:]
        else:
            data_slice = data[i*ticks_per_iteration:(i+1)*ticks_per_iteration]

        # fix missing value issue by inserting a 0 at time to be picked up later
        # convert datetime to timestamps

        ret      = np.divide(data_slice['price'][1:], data_slice['price'][:-1]) - 1
        p_values.append(stats.normaltest(ret)[1])

    return p_values

def scrub_data(rank, data, ticks_per_iteration, ret_threshold=0.5):

    iterations       = int(np.ceil(float(len(data)) / ticks_per_iteration))
    bad_data_records = ''

    for i in range(iterations):

        if i == (iterations-1):
            data_slice = data[i*ticks_per_iteration:]
        else:
            data_slice = data[i*ticks_per_iteration:(i+1)*ticks_per_iteration]

        # fix missing value issue by inserting a 0 at time to be picked up later
        # convert datetime to timestamps

        data_slice = np.sort(data_slice, order='time')
        ret        = np.divide(data_slice['price'][1:], data_slice['price'][:-1]) - 1
        time_drift = data_slice['time'][1:] - data_slice['time'][:-1]
        time_mean  = data_slice['time'].mean()
        time_std   = data_slice['time'].std()

        d = (data_slice['price'] <= 0)
        e = (data_slice['volume']<= 0)
        f = (data_slice['time']  == 0) | (data_slice['time'] > time_mean + 10 * time_std) | (data_slice['time'] < time_mean - 10 * time_std)
        g = (ret > ret_threshold) | (ret < (-1) *ret_threshold)
        h = (time_drift == 0)

        data_slice['flag'][d] = 1
        data_slice['flag'][e] = 2
        data_slice['flag'][f] = 3
        data_slice['flag'][g] = 4
        data_slice['flag'][h] = 5

        for record in data_slice[data_slice['flag'] != 0]:
            flag     = BAD_DATA_REASON_MAP[str(record['flag'])]
            record   = list(record.reshape(-1)[['index', 'timestamp', 'price', 'volume']].tolist()[0])
            record   = [str(x) for x in record]
            bad_data = ','.join(record)
            bad_data = bad_data + ',' + flag + '\n'
            bad_data_records += bad_data

    return bad_data_records

def write_data(output_file, records):
    output_file.Write_ordered(records.encode())
