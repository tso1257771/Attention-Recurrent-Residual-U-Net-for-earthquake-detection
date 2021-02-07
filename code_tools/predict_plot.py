import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12
from obspy.signal import filter
from scipy.signal import tukey
from .data_utils import pick_peaks

def plot_pred_by_index(idx, X, Y, predictions, search_win=1, 
            draw_hp_wf=None, hp_freq=2, dt=0.01, snr=None, save=None):
    x=X[idx].T
    y=Y[idx].T
    predict = predictions[idx].T
    length = len(predict[0])
    if draw_hp_wf:
        x = [ tukey(length)*(filter.highpass(x[i], 
                        freq=hp_freq, df=1/dt)) for i in range(3)]
    
    x_plot = [x[0], x[1], x[2], predict[0], predict[1], predict[2]]
    label = ["E comp.", "N comp.", "Z comp.", 
            "P prob.", "S prob.", 'Noise prob']

    P_idx = np.argmax(y[0])
    S_idx = np.argmax(y[1]) 
    # pick from predictions
    pickP, probP = pick_peaks(predict[0], P_idx*dt,
                     sac_dt=dt, search_win=search_win)
    pickP_idx = int(pickP/dt) if pickP != -999  else None

    pickS, probS = pick_peaks(predict[1], S_idx*dt,
                     sac_dt=dt, search_win=search_win)
    pickS_idx = int(pickS/dt) if pickS != -999  else None

    try:
        if snr.any():
            p_snr = snr[idx]
    except:
        p_snr=None

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1,
                                 sharex=True, figsize=(8, 8))
    ax=[ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(6):
        ax[i].plot(x_plot[i])
        if 2 >= i:
            if np.max(y[0])>0 and np.max(y[1])>0:   
                #ax[i].axvline(x=P_idx, label='Manual picked P', color='k')
                #ax[i].axvline(x=S_idx, label='Manual picked S ', color='r')
                ax[i].axvline(x=P_idx, label='Inverse P', color='k', linestyle=':')
                ax[i].axvline(x=S_idx, label='Inverse S ', color='r', linestyle=':')                
        if 4 >= i:
            if pickP_idx: 
                if i != 4:
                    ax[i].axvline(x=pickP_idx, 
                        label='Predicted P', color='k', linestyle='-')
                if i == 3:    
                    ax[i].text(pickP_idx+(0.01*length), 
                        np.max(x_plot[i])*0.5, f'P prob. {probP:.2f}')
        
            if pickS_idx:
                if i != 3:
                    if not pickP_idx==None:
                        ax[i].axvline(x=pickS_idx, label='Predicted S', 
                            color='r', linestyle='-')
                if i == 4:    
                    if not pickS_idx==None:
                        ax[i].text(pickS_idx+(0.01*length), np.max(x_plot[i])*0.5,
                            f'S prob. {probS:.2f}')

        ax[i].set_ylabel(label[i])
    ax[-1].set_xlabel('samples')
    if snr:
        ax[0].set_title(f'High-passed SNR: {p_snr:.2f}')
    ax[0].legend(loc='upper right')

    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_detect_by_index(trc, label, mask, predictions,
        idx, draw_hp_wf=2, data_length=2001, 
        dt=0.01, search_win=1, save=None):
    trc = trc.numpy()
    label = label.numpy()
    mask = mask.numpy()

    Trc = trc[idx].T
    Label = label[idx].T
    Mask = mask[idx].T
    pred_label = predictions[0][idx].T
    pred_mask = predictions[1][idx].T

    if draw_hp_wf:
        Trc = [ tukey(data_length, alpha=0.1)*(filter.highpass(Trc[i], 
                        freq=draw_hp_wf, df=1/dt)) for i in range(3)]
    
    x_plot = [Trc[0], Trc[1], Trc[2], 
            pred_label[0], pred_label[1], pred_mask[0]]
    label = ["E comp.", "N comp.", "Z comp.", 
            "P prob.", "S prob.", 'EQ mask\nprob.']

    P_idx = np.where(Label[0]==1)[0]
    S_idx = np.where(Label[1]==1)[0]
    # pick from predictions
    pickP_idxs = []
    probPs = []
    pickS_idxs = []
    probSs = []    
    for p in range(len(P_idx)):
        pickP, probP = pick_peaks(pred_label[0], P_idx[p]*dt,
                        sac_dt=dt, search_win=search_win)
        pickP_idx = int(pickP/dt) if pickP != -999  else None
        pickP_idxs.append(pickP_idx)
        probPs.append(probP)
    for s in range(len(S_idx)):
        pickS, probS = pick_peaks(pred_label[1], S_idx[s]*dt,
                        sac_dt=dt, search_win=search_win)
        pickS_idx = int(pickS/dt) if pickS != -999  else None
        pickS_idxs.append(pickS_idx)
        probSs.append(probS)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1,
                                 sharex=True, figsize=(8, 8))
    ax=[ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(6):
        ax[i].plot(x_plot[i])
        ax[i].set_xlim(0, data_length)
        if 2 >= i:
            if np.any(P_idx):
                for p in range(len(P_idx)):         
                    ax[i].axvline(x=P_idx[p], 
                        label='Manual picked P',
                        color='k', linestyle=':')
            if np.any(S_idx):            
                for s in range(len(S_idx)):        
                    ax[i].axvline(x=S_idx[s],
                        label='Manual picked S ', 
                        color='r', linestyle=':')          
        if 4 >= i:
            if np.any(pickP_idxs): 
                if i != 4:
                    for p in range(len(pickP_idxs)):
                        if not pickP_idxs[p]==None:
                            ax[i].axvline(x=pickP_idxs[p], 
                            label='Predicted P', 
                            color='k', linestyle='-')
                    if i == 3:
                        if not pickP_idxs[p]==None:
                            ax[i].text(pickP_idxs[p]+(dt*data_length), 
                                np.max(x_plot[i])*0.5, 
                                f'P prob. {probPs[p]:.2f}')
        
            if np.any(pickS_idxs): 
                if i != 3:
                    for s in range(len(pickS_idxs)):
                        if not pickS_idxs[s]==None:
                            ax[i].axvline(x=pickS_idxs[s], 
                            label='Predicted S',
                            color='r', linestyle='-')
                    if i == 4:
                        if not pickS_idxs[s]==None:
                            ax[i].text(pickS_idxs[s]+(dt*data_length), 
                                np.max(x_plot[i])*0.5,
                                f'S prob. {probSs[s]:.2f}')

        ax[i].set_ylabel(label[i])
    ax[-1].set_xlabel('Data samples')
    ax[0].legend(loc='upper right')
    fig.align_ylabels(ax[:])
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_pred_single(wf_data, pred_data, data_info, 
        sac_info, label=None, search_win=1, hp_freq=2, save=None):

    dt = sac_info.delta
    npts = sac_info.npts
    tp = int(round(sac_info.t1/dt))
    ts = int(round(sac_info.t2/dt))
    pickP = int(round(data_info['predP']/dt))
    pickS = int(round(data_info['predS']/dt))

    #raw_Psnr = float(sac_info.kuser0)
    if hp_freq:
        wf_data[0] = tukey(npts)*\
            (filter.highpass(wf_data[0], freq=hp_freq, df=1/dt))
        wf_data[1] = tukey(npts)*\
            (filter.highpass(wf_data[1], freq=hp_freq, df=1/dt))
        wf_data[2] = tukey(npts)*\
            (filter.highpass(wf_data[2], freq=hp_freq, df=1/dt))
    x_plot = [wf_data[0], wf_data[1], wf_data[2],
             pred_data[0], pred_data[1], pred_data[2]]
    if not label:
        label = ["E comp.", "N comp.", "Z comp.",
                 "P prob.", "S prob.", 'Noise prob']
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = \
        plt.subplots(6, 1, sharex=True, figsize=(8, 8))
    ax=[ax1, ax2, ax3, ax4, ax5, ax6]

    for i in range(6):
        ax[i].plot(x_plot[i], linewidth=1)
        if 2 >= i: 
            ax[i].axvspan(tp-search_win/dt, tp+search_win/dt,
                 facecolor='green', alpha=0.2)
            ax[i].axvspan(ts-search_win/dt, ts+search_win/dt,
                 facecolor='gray', alpha=0.2)
            ax[i].axvline(x=tp, label='Manual picked P',
                 color='k', linestyle=':', linewidth=2)
            ax[i].axvline(x=ts, label='Manual picked S',
                 color='r', linestyle=':', linewidth=2)            
            if pickP>0:
                ax[i].axvline(x=pickP, label='Predicted P',
                 color='k', linestyle='-')
            if pickS>0:
                ax[i].axvline(x=pickS, label='Predicted S ',
                 color='r', linestyle='-')
        ax[i].set_ylabel(label[i])
    #ax[0].set_title(f'Raw high-passed SNR: {raw_Psnr:.2f}')
    ax[0].legend(loc='upper right')
    if save:
        message = f'Figure saved to {save}'
        plt.savefig(save)
        plt.close()
        return message
    else:
        message = f'Plotting ...'
        return message
        plt.show()

def plot_mosaic_pred_multiple(trc, predict, p_ans, s_ans, p_peaks, s_peaks, p_values, s_values,
                                search_win=1, dt=0.01, hp_freq=None, save=None):
    npts = len(predict[0])
    #raw_Psnr = float(sac_info.kuser0)
    if hp_freq:
        trc[0] = tukey(npts)*(filter.highpass(trc[0], freq=hp_freq, df=1/dt))
        trc[1] = tukey(npts)*(filter.highpass(trc[1], freq=hp_freq, df=1/dt))
        trc[2] = tukey(npts)*(filter.highpass(trc[2], freq=hp_freq, df=1/dt))
    x_plot = [trc[0], trc[1], trc[2], predict[0], predict[1], predict[2]]
    label = ["E comp.", "N comp.", "Z comp.", "P prob.", "S prob.", 'Noise prob']
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(8, 8))
    ax=[ax1, ax2, ax3, ax4, ax5, ax6]

    for i in range(6):
        ax[i].plot(np.arange(npts), x_plot[i], linewidth=1)
        ax[i].set_xlim(0, npts)
        if 2 >= i: 
            for j1 in range(len(p_ans)):
                ax[i].axvspan((p_ans[j1]-search_win)/dt, (p_ans[j1]+search_win)/dt, facecolor='green', alpha=0.2)
                if j1 == 0:
                    ax[i].axvline(x=int(p_ans[j1]/dt), label='Manual picked P', color='k')
                else:
                    ax[i].axvline(x=int(p_ans[j1]/dt), color='k')
            for j2 in range(len(s_ans)):
                ax[i].axvspan((s_ans[j2]-search_win)/dt, (s_ans[j2]+search_win)/dt, facecolor='gray', alpha=0.2)
                if j2 == 1:
                    ax[i].axvline(x=int(s_ans[j2]/dt), label='Manual picked S', color='r')
                else:
                    ax[i].axvline(x=int(s_ans[j2]/dt), color='r')

        # put on predicted value
        for k1 in range(len(p_peaks)):
            if p_peaks[k1] > 0:
                if i in (0, 1, 2, 3):
                    ax[i].axvline(x=int(p_peaks[k1]/dt), color='k', linestyle=':')
                if i == 3:
                    ax[i].text( int(p_peaks[k1]/dt), p_values[k1], f'{p_values[k1]:.2f}')
        for k2 in range(len(s_peaks)):
            if s_peaks[k2]>0:
                if i in (0, 1, 2, 4):
                    ax[i].axvline(x=int(s_peaks[k2]/dt), color='r', linestyle=':')
                if i == 4:
                    ax[i].text( int(s_peaks[k2]/dt), s_values[k2], f'{s_values[k2]:.2f}')       
        ax[i].set_ylabel(label[i])
    #ax[0].set_title(f'Raw high-passed SNR: {raw_Psnr:.2f}')
    ax[0].legend(loc='upper right')
    ax[5].set_xlabel('Time (npts)')
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    if save:
        message = f'Figure saved to {save}'
        plt.savefig(save)
        plt.close()
        return message
    else:
        message = f'Plotting ...'
        return message
        plt.show()