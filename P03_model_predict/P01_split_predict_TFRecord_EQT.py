import os
import h5py
import logging
import sys
sys.path.append('../')
import obspy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from obspy import read
from obspy.io.sac.sactrace import SACTrace
from scipy.signal import find_peaks
from code_tools.model.build_model import unets
from code_tools.data_utils import pick_peaks, snr_pt_v2
from code_tools.data_io import tfrecord_dataset_detect
from tensorflow.keras.models import load_model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s : %(asctime)s : %(message)s')

basepath = '../'
mdl_hdr = 'ARRU_detect/EQT_20s'
TFRpath = os.path.join(basepath, 'STEAD/input_TFRecord_EQT_20s')

#datapath = os.path.join(basepath, 'STEAD/sac_data_EQT_20s/test')
#pred_list = np.random.permutation(glob(
#    os.path.join(TFRpath, 'test/*.tfrecord')))
#outpath = os.path.join(basepath, 
#            f's03_test_predict/{mdl_hdr}_on_STEAD')
datapath = os.path.join(basepath, 'STEAD/sac_data_EQT_20s/val')
pred_list = np.random.permutation(glob(
    os.path.join(TFRpath, 'val/*.tfrecord')))
outpath = os.path.join(basepath, 
            f's03_test_predict/{mdl_hdr}_on_STEAD_val')

data_length = 2001
frame = unets(input_size=(data_length, 3))
mdl_dir = os.path.join(basepath, 
    's02_trained_model', mdl_hdr) #166, 241, 208
model = frame.build_attR2unet(
    os.path.join(mdl_dir, 'train.hdf5'), 
    input_size=(data_length, 3))
#model = load_model(os.path.join(mdl_dir, 'train.hdf5'))

use_aicP = False
use_aicS = False
use_hp = None
calculate_snr = False
p_snr_win = 0 #seconds
os.system(f'rm -rf {outpath}')
if not os.path.exists(outpath):
    os.makedirs(outpath)
message_out = os.path.join(outpath, 'summary.txt')
f_out = open(message_out, 'w')
f_out.write('        evid,   sta, chn,  |manP-manS|'+\
            ', manP,  manS, predP,'
            ' predP_prob, predS, predS_prob,'+\
            '       dist, hp_Psnr, hp_Ssnr\n')

split_iter = 100
sep_idx = int(len(pred_list)//split_iter)
ct=0
for i in range(split_iter):
    st_idx, end_idx = sep_idx*i, sep_idx*(i+1)
    if i != split_iter-1:
        pred_batch_list = pred_list[st_idx:end_idx]
        n_pred = len(pred_batch_list)
        pred_batch = iter(tfrecord_dataset_detect(pred_batch_list,
            repeat=1, data_length=data_length, 
            batch_size=n_pred, shuffle_buffer_size=300))
    else:
        pred_batch_list = pred_list[st_idx:]
        n_pred = len(pred_batch_list)
        pred_batch = iter(tfrecord_dataset_detect(pred_batch_list, 
            repeat=1, data_length=data_length,
            batch_size=n_pred, shuffle_buffer_size=300))
    pred_trc, _, _, idx = next(pred_batch)

    logging.info(f"Making predictions ... : {i+1}/{split_iter}")
    predictions = model.predict(pred_trc)

    for k in range(n_pred):
        logging.info(f"Generating predictions ... "+\
            f"{ct+1}/{len(pred_list)}: {idx[k].numpy().decode('utf-8')}")
        trc_id_decode = idx[k].numpy().decode('utf-8').split('.')
        evid = trc_id_decode[0]
        chn = trc_id_decode[1]
        sta_info = '.'.join(trc_id_decode[2:])

        predict = predictions[k].T
        predict_P = predict[0]
        predict_S = predict[1]
        predict_nz = predict[2]

        if use_hp:
            sac = read(os.path.join(datapath, evid,
                    f'{evid}.{sta_info}.{chn}.sac.norm.hp'))
            sac.filter('highpass', freq=use_hp)
            sac.taper(max_percentage=0.05)
            sac.detrend('demean')
            for s in sac:
                s.data/=np.std(s.data)
            sac.sort()
            sac_E = sac[0].copy()
            sac_N = sac[1].copy()
            sac_Z = sac[2].copy()        
        else:
            sac = read(os.path.join(datapath, evid, 
                f'{evid}.{sta_info}.{chn}.sac.norm'), headonly=True)

        sac_info = sac[0].stats.sac
        sac_dt = sac_info.delta

        if use_aicS:
            ts_utc = sac[0].stats.starttime+sac_info.t6
            labeled_S = sac_info.t6
            ts_to_end = (data_length-1)*sac_dt-sac_info.t6
        elif not use_aicS:
            ts_utc = sac[0].stats.starttime+sac_info.t2
            labeled_S = sac_info.t2
            ts_to_end = (data_length-1)*sac_dt-sac_info.t2

        if use_aicP:
            tp_utc = sac[0].stats.starttime+sac_info.t5
            labeled_P = sac_info.t5
            tp_to_end = (data_length-1)*sac_dt-sac_info.t5
        elif not use_aicP:
            tp_utc = sac[0].stats.starttime+sac_info.t1
            labeled_P = sac_info.t1
            tp_to_end = (data_length-1)*sac_dt-sac_info.t1

        dist = -999
        raw_tp_ts_diff = labeled_S-labeled_P
        p_peak, p_value = pick_peaks(predict_P, 
                labeled_P, sac_dt, search_win=1)
        s_peak, s_value = pick_peaks(predict_S, 
                labeled_S, sac_dt, search_win=1)

        if calculate_snr:
            sac_E = sac[0].copy()
            sac_N = sac[1].copy()
            sac_Z = sac[2].copy()
            hp_p_snr, hp_s_snr = snr_pt_v2(tr_vertical=sac_Z,
                tr_horizontal=sac_E, pt_p=tp_utc, 
                pt_s=ts_utc, mode='sqrt',
                snr_pre_window=p_snr_win, 
                snr_post_window=p_snr_win, highpass=2)
        else:
            #hp_p_snr, hp_s_snr = float(sac_info.kuser0.strip()),\
            #    float(sac_info.kuser1.strip())
            hp_p_snr, hp_s_snr = -999, -999

        message = f"{evid}, {sta_info:>5s}, {chn:>3s}, "+\
                  f"{raw_tp_ts_diff:>11.2f}, "+\
                  f"{labeled_P:>5.2f}, {labeled_S:>5.2f}, "+\
                  f"{p_peak:>5.2f}, {p_value:>10.2f}, "+\
                  f"{s_peak:>5.2f}, {s_value:>10.2f}, {dist:>10.2f}, "+\
                  f"{hp_p_snr:8.2f}, {hp_s_snr:8.2f}\n"
        f_out.write(message)
        ct += 1
logging.info(f"Predicted samples : {ct}")
logging.info(f"Summary file: {message_out}")
f_out.close()

