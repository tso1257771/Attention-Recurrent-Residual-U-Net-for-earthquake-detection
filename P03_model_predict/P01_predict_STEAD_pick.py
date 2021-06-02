import os
import logging
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import load_model
from code_tools.model.build_model import unets
from code_tools.data_utils import pick_peaks
from code_tools.data_io import tfrecord_dataset_detect

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
        format='%(levelname)s : %(asctime)s : %(message)s')

basepath = '../'
mdl_dir = os.path.join(basepath, 
    'pretrained_model/paper_model_ARRU_20s')
TFRpath = os.path.join(basepath, 
    'data/input_TFRecord_STEAD_20s')
datapath = os.path.join(basepath, 
    'STEAD/sac_data_STEAD_20s/test')
pred_list = np.random.permutation(glob(
    os.path.join(TFRpath, 'test/*.tfrecord')))
outpath = os.path.join(basepath, 
    f's03_test_predict/{os.path.basename(mdl_dir)}_on_STEAD')

dt = 0.01
data_length = 2001
plot_fig = 100 # or set 'False'

## load model from model architecture
frame = unets(input_size=(data_length, 3))
model = frame.build_attR2unet(
    os.path.join(mdl_dir, 'train.hdf5'), 
    input_size=(data_length, 3))
## or just load hdf5 file
#model = load_model(os.path.join(mdl_dir, 'train.hdf5'))

os.system(f'rm -rf {outpath}')
if not os.path.exists(outpath):
    os.makedirs(outpath)
fig_path = os.path.join(outpath, 'fig')
if plot_fig:
    os.system(f'rm -rf {fig_path}')
    os.makedirs(fig_path)

message_out = os.path.join(outpath, 'summary.txt')
f_out = open(message_out, 'w')
f_out.write(' evid, sta, chn, |manP-manS|, '+\
    'manP, manS, predP, predP_prob, predS, predS_prob,'+\
    'real_type, pred_type, eq_prob\n')
split_iter = 10
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
    pred_trc, label, mask, idx = next(pred_batch)

    logging.info(f"Making predictions ... : {i+1}/{split_iter}")
    predictions = model.predict(pred_trc)

    for k in range(n_pred):
        logging.info(f"Generating predictions ... "+\
            f"{ct+1}/{len(pred_list)}: {idx[k].numpy().decode('utf-8')}")
        trc_id_decode = idx[k].numpy().decode('utf-8').split('.')
        evid = trc_id_decode[0]
        chn = trc_id_decode[1]
        sta_info = '.'.join(trc_id_decode[2:])

        # labeled P&S arrival time
        label_info = label[k].numpy()
        try:
            labeled_P = np.where(label_info.T[0]==1)[0][0]*dt
            labeled_S = np.where(label_info.T[1]==1)[0][0]*dt
            raw_tp_ts_diff = labeled_S-labeled_P
        except: 
            # cannot find labeled P/S, 
            # representing the pure noise seismogram
            continue

        # predictions information
        predict = predictions[k].T
        predict_P = predict[0]
        predict_S = predict[1]
        predict_nz = predict[2]
        p_peak, p_value = pick_peaks(predict_P, 
                labeled_P, 0.01, search_win=1)
        s_peak, s_value = pick_peaks(predict_S, 
                labeled_S, 0.01, search_win=1)

        message = f"{evid}, {sta_info:>10s}, {chn:>3s}, "+\
                  f"{raw_tp_ts_diff:>11.2f}, "+\
                  f"{labeled_P:>5.2f}, {labeled_S:>5.2f}, "+\
                  f"{p_peak:>5.2f}, {p_value:>10.2f}, "+\
                  f"{s_peak:>5.2f}, {s_value:>10.2f}, "+\
                  "earthquake, -999, -999\n"
        f_out.write(message)

        if plot_fig and ct < plot_fig:
            fig, ax = plt.subplots(5, 1, figsize=(10, 6))
            xs = [pred_trc[k].numpy().T[0],
                  pred_trc[k].numpy().T[1],
                  pred_trc[k].numpy().T[2],
                  predict_P,
                  predict_S]
            x_lbl = ['E', 'N', 'Z', 'P', 'S']
            for x in range(5):
                ax[x].plot(xs[x], linewidth=1)
                ax[x].set_ylabel(x_lbl[x])
                ax[x].set_xlim(0, data_length)


                if x != 4:
                    ax[x].axvline(np.round(p_peak/dt).astype(int), 
                        linewidth=1, linestyle='-', 
                        color='k', label='predicted P')
                    if p_value > 0.3:
                        ax[x].axvline(np.round(labeled_P/dt).astype(int), 
                            linewidth=1, linestyle=':', 
                            color='k', label='labeled P')
                if x != 3:
                    ax[x].axvline(np.round(s_peak/dt).astype(int), 
                        linewidth=1, linestyle='-', 
                        color='r', label='predicted S')
                    if s_value > 0.3:
                        ax[x].axvline(np.round(labeled_S/dt).astype(int), 
                            linewidth=1, linestyle=':', 
                            color='r', label='labeled S')
                if x > 2:
                    ax[x].set_ylim(-0.1, 1.1)
                if x < 4:
                    ax[x].set_xticks([])
            ax[4].set_xlabel('npts')
            ax[3].legend(); ax[4].legend()
            fig_id = f'pred_{evid}.{chn[:2]}{sta_info}.png'
            plt.savefig(os.path.join(fig_path, fig_id))
            plt.close()
        ct += 1

logging.info(f"Predicted earthquake samples : {ct}")
logging.info(f"Summary file: {message_out}")
f_out.close()

