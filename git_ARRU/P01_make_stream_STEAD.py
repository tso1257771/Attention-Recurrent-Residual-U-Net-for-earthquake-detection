import os
import sys
sys.path.append('../')
import h5py
import obspy
import logging
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
from obspy.io.sac.sactrace import SACTrace
from STEAD_tools.make_sac import stream_from_h5
from STEAD_tools.STEAD_utils import *
from code_tools.data_utils import snr_pt_v2
from code_tools.data_utils import stream_standardize
from code_tools.data_utils import gen_tar_func
from code_tools.example_parser import write_TFRecord_detect

logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

h5 = './merge.hdf5'
csv_type = 'test'
outdir = os.path.join(f'./data/sac_data_STEAD_20s/{csv_type}')
outtf = os.path.join(f'./data/input_TFRecord_STEAD_20s/{csv_type}')
os.system(f'rm -rf {outdir} {outtf}')
data_csv = f'./out_csv/{csv_type}_STEAD.csv'
if not os.path.exists(outtf):
    os.makedirs(outtf)

# read h5py file
dtfl = STEAD_h5(h5)

calculate_snr = True
snr_win = 3
bandpass = [1, 45]
data_length = 2001
data_sec = 20
dt = 0.01
err_win_p = 0.4
err_win_s = 0.4
secs_bef_P = 1
secs_aft_S = 3
ps_res_limit = 15.9
# convering hdf5 dataset into obspy sream
tf_ct = 0
sac_ct = 0
# get information
csv_data = STEAD_csv(data_csv)
df_cat = csv_data.trace_category.values
df_evid = csv_data.trace_name.values

rej = 1
for ct, p in enumerate(df_evid):
    logging.info(f"Processing: {ct+1}/{len(df_evid)}: {p}")
    # retreive data from hdf5 
    data_st = dtfl.get(f'data/{p}')
    sta_info, evid = p.split('_')[:2]

    # set up output directory
    outpath = os.path.join(outdir, evid)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # make temporary sac file
    _st, _tp, _ts, ps_res_npts = stream_from_h5(data_st)
    for _sac in _st:
        chn = _sac.stats.channel
        out_sac_temp = os.path.join(outpath, 
            f'{evid}.{sta_info}.{chn}.sac.norm')
        _sac.write(out_sac_temp, format='SAC')
        #print(out_sac_temp)
    read_idx = out_sac_temp.replace('Z.sac.norm', '?.sac.norm')
    st = read(read_idx)

    # specify data category
    is_noise = np.logical_and(_tp==-999, _ts==-999)

    # define P/S arrival time and SNR in UTCDateTime format
    if not is_noise:
        tp_utc = st[0].stats.starttime + _tp
        ts_utc = st[0].stats.starttime + _ts
        # neglect data with :
        # 1. |ts-tp| > ps_res_limit
        # 2. length before P < secs_bef_P
        # 3. length after S < secs_aft_S
        if (ts_utc-tp_utc) > ps_res_limit : #or\
            #   _tp < secs_bef_P or (data_sec - _ts) < secs_aft_S:
            continue
        if (calculate_snr and _tp > snr_win):
            hp_p_snr, hp_s_snr = snr_pt_v2(
                st[2], st[1], tp_utc, ts_utc, mode='std', 
                snr_pre_window=snr_win, snr_post_window=snr_win, highpass=2)
        elif not (calculate_snr and _tp > snr_win):
            hp_p_snr, hp_s_snr = -999, -999
    elif is_noise:
        tp_utc, ts_utc, hp_p_snr, hp_s_snr = -999, -999, -999, -999

    # randomly assign initial starttime
    if is_noise:
        P_prewin = _tp
    elif np.logical_and(is_noise==False, hp_p_snr==-999):
        P_prewin = _tp
    elif np.logical_and(is_noise==False, hp_p_snr!=-999):
        P_prewin = _tp
        loop_ct = 0
        while P_prewin >= _tp:
            P_prewin = secs_bef_P + np.random.choice(
                int(data_length - ps_res_npts - 
                    secs_bef_P/dt - secs_aft_S/dt - 1))*dt
            loop_ct += 1
            if loop_ct > 10000:
                break

    # slice the trace
    if P_prewin == -999:
        slice_stt = st[0].stats.starttime
    else:
        slice_stt = tp_utc - P_prewin
    slice_ent = slice_stt + data_sec
    slice_st = st.copy().slice(slice_stt, slice_ent)
    # return slice_st, tp_utc, ts_utc, hp_p_snr, hp_s_snr, is_noise

    # check data length
    data_len = [len(i.data) for i in slice_st]
    check_len = np.array_equal(data_len, np.repeat(data_length, 3))
    if not check_len:
        for s in slice_st:
            res_len = len(s.data) - data_length
            if res_len > 0:
                s.data = s.data[:data_length]
            elif res_len < 0:
                s.data = np.insert(s.data, -1, np.zeros(res_len))
    #if not check_len:
    #    stop
    if bandpass:
        slice_st= slice_st.detrend('demean').filter(
            'bandpass', freqmin=bandpass[0], freqmax=bandpass[1])
    new_st = stream_standardize(slice_st)

    # check data infinity/nan
    check_trc = [i.data for i in new_st]
    if np.logical_or(np.isnan(check_trc).any(), np.isinf(check_trc).any()):
        continue

    # make new sac header
    new_sac_dict = new_st[0].stats.sac.copy()
    if df_cat[ct] != 'noise':
        new_sac_dict['t1'] = tp_utc - new_st[0].stats.starttime
        new_sac_dict['t2'] = ts_utc - new_st[0].stats.starttime
    else:
        new_sac_dict['t1'] = -999
        new_sac_dict['t2'] = -999
    new_sac_dict['kuser1'] = str(hp_p_snr)
    new_sac_dict['kuser2'] = str(hp_s_snr)

    # make new sac
    for s in new_st:
        s.stats.sac = obspy.core.AttribDict(new_sac_dict)

        chn = s.stats.channel      
        net = s.stats.network      
        out_norm = f'{evid}.{sta_info}.{chn}.sac.norm'
        #print(os.path.join(outpath, out_norm))

        wf = SACTrace.from_obspy_trace(s)
        wf.b = 0
        wf.write(os.path.join(outpath, out_norm))
    sac_ct += 1
    new_st.sort()

    # make tfrecord
    try:
        tp_npts =  int(new_sac_dict['t1']/dt)
        ts_npts =  int(new_sac_dict['t2']/dt)
        trc_E = new_st[0].data
        trc_N = new_st[1].data
        trc_Z = new_st[2].data  

        if df_cat[ct] != 'noise':
            # target function for phase picking
            trc_tp = gen_tar_func(data_length, tp_npts, int(err_win_p/dt)+1)
            trc_ts = gen_tar_func(data_length, ts_npts, int(err_win_s/dt)+1)
            trc_tn = np.ones(data_length) - trc_tp - trc_ts
            # target function for phase arrival masks
            trc_mask = trc_tp+trc_ts
            trc_mask[tp_npts:ts_npts+1] = 1
            trc_unmask = np.ones(data_length) - trc_mask
        else:
            trc_tp = np.zeros(data_length)
            trc_ts = np.zeros(data_length)
            trc_tn = np.ones(data_length)
            trc_mask = np.zeros(data_length)
            trc_unmask = np.ones(data_length)

        # reshape for input U net model
        trc_3C = np.array([trc_E, trc_N, trc_Z]).T
        label_psn = np.array([trc_tp, trc_ts, trc_tn]).T
        mask = np.array([trc_mask, trc_unmask]).T

        if np.logical_or(np.isinf(trc_3C).any(), 
                np.isnan(trc_3C).any()):
            raise ValueError
    except:
        stop
        continue

    idx = f'{evid}.{chn[:2]}?.{sta_info}'
    out_tfid = f'{evid}.{sta_info}.{chn[:2]}.tfrecord'
    outfile = os.path.join(outtf, out_tfid)
    write_TFRecord_detect(trc_3C, label_psn, mask, 
                idx=idx, outfile=outfile)
    tf_ct += 1
logging.info(f"{outtf}: {tf_ct}")