import obspy
import numpy as np
from obspy import read, UTCDateTime

def stream_from_h5(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream

def zero_pad_stream(slice_st, data_length, zero_pad_range,
        max_pad_slices=4, random_chn=True):
    '''
    Randomly pad the noise waveform with zero values on channels

    '''

    # 0 for padding with zero; 1 for no padding
    if random_chn:
        pad_chn = np.array([np.random.randint(2) for i in range(3)])
        pad_chn_idx = np.where(pad_chn==1)[0]
        if pad_chn_idx.sum()==0:
            pad_chn_idx = np.array([0, 1, 2])
    else:
        pad_chn_idx = np.array([0, 1, 2])

    zero_pad = np.random.randint(
        zero_pad_range[0], zero_pad_range[1])
    max_pad_seq_num = np.random.randint(max_pad_slices)+1
    pad_len = np.random.multinomial(zero_pad, 
        np.ones(max_pad_seq_num)/max_pad_seq_num)

    for ins in range(len(pad_len)):
        max_idx = data_length - pad_len[ins]
        insert_idx = np.random.randint(max_idx)
        for ch in pad_chn_idx:
            slice_st[ch].data[
                insert_idx:(insert_idx+pad_len[ins])] = 0
    return slice_st