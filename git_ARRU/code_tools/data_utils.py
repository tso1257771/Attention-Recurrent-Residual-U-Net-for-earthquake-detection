import numpy as np
from obspy import read
from obspy.signal import filter
from scipy.signal import tukey
from obspy.signal.filter import envelope
from scipy.signal import find_peaks

def gen_tar_func(data_length, point, mask_window):
    '''
    data_length: target function length
    point: point of phase arrival
    mask_window: length of mask, must be even number
                 (mask_window//2+1+mask_window//2)
    '''
    target = np.zeros(data_length)
    half_win = (mask_window-1)//2
    gaus = np.exp(-(
        np.arange(-half_win, half_win+1))**2 / (2*(half_win//2)**2))
    #print(gaus.std())
    gaus_first_half = gaus[:mask_window//2]
    gaus_second_half = gaus[mask_window//2+1:]
    target[point] = gaus.max()
    #print(gaus.max())
    if point < half_win:
        reduce_pts = half_win-point
        start_pt = 0
        gaus_first_half = gaus_first_half[reduce_pts:]
    else:
        start_pt = point-half_win
    target[start_pt:point] = gaus_first_half
    target[point+1:point+half_win+1] = \
        gaus_second_half[:len(target[point+1:point+half_win+1])]
    return target

def BP_envelope_eq_info(glob_list, freqmin=1, freqmax=20,
        p_hdr='t5', s_hdr='t6', std_pt_bef_P=0.5, mva_win_sec=3,
        env_end_ratio=0.8):
    '''Define an earthquake event using filtered envelope function
    and its moving average.
    # --------- input
    * read_list: list of event files
    * freqmin / freqmax: corner frequency for band-pass filtering
    * p_hdr / s_hdr: index of labeled P/S phase stored in sac header
    * std_pt_ber_P: standard check point for moving average value
            of the envelope function before labeled P arrival (secs)
    * mva_win_sec: window length for calculating moving average (secs)
    * env_end_ratio: ratio of event end point and standard point on 
            envelope function.
    # -------- output
    * tp_pt / ts_pt 
    * dt
    * st_env
    * env_mva
    * env_standard_pt 
    * end_ev_pt
    '''
    st = read(glob_list)
    st.sort()
    st = st.detrend('demean')
    for s in st:
        s.data /= np.std(s.data)
    st = st.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    info = st[0].stats
    dt = info.delta
    tp_pt = int(round((info.sac[p_hdr] - info.sac.b)/dt, 2))
    ts_pt = int(round((info.sac[s_hdr] - info.sac.b)/dt, 2))

    env_standard_pt = int(round((
        info.sac.t5 - info.sac.b - std_pt_bef_P)/dt, 2))

    # establish moving average function of filtered envelope function
    arr_shape = len(st), len(st[0].data)
    mva_win = int(mva_win_sec/dt)
    st_env = np.zeros(arr_shape)
    env_mva = np.zeros(arr_shape)
    for ct, p in enumerate(st):
        st_env[ct] = envelope(p.data)
        mva = np.convolve(st_env[0],
            np.ones((mva_win,))/mva_win, mode='valid')
        env_mva[ct][-len(mva):] = mva
    # find event end point across channels
    end_ev_pts = np.zeros(2)
    for i in range(len(st)-1):
        end_v_search = np.where(
            env_mva[i] <= env_mva[i][env_standard_pt]/env_end_ratio )[0]
        end_ev_pts[i] = end_v_search[ 
            np.where(end_v_search-ts_pt > 0)[0][0] ]
    end_ev_pt = int(np.mean(end_ev_pts))

    return tp_pt, ts_pt, dt, st_env, env_mva, env_standard_pt, end_ev_pt

def assign_slice_window(p_s_residual, data_length, 
                        avail_bef_P, avail_aft_S, dt):
    """
    `p_s_residual`: |P_arrival - S_arrival|
    `data_length`: total length of sliced waveform
    `avail_bef_P`: available dataspace before P phase
    `avail_aft_S`: available dataspace ater S phase

    Conditioning
    -P_prewin >= avail_bef_P
    -S_prewin = P_prewin + p_s_residual
    -(S_prewin + avail_aft_S ) < data_length

    P_prewin: length of time window before P arrival
    return P_prewin
    
    """
    avail_bef_P/=dt
    avail_aft_S/=dt

    P_avail_space = np.arange(avail_bef_P, 
                (data_length - p_s_residual - avail_aft_S), 1)
    P_prewin = np.random.choice(P_avail_space)
    return P_prewin

def snr_pt(tr, pt, mode='std',
            snr_pre_window=5, snr_post_window=5):
    """
    Calculate snr
    tr: sac trace
    pt: utcdatetime object
    """
    tr_s = tr.copy()
    tr_n = tr.copy()
  
    if mode.lower() == 'std':
        tr_noise = np.std(tr_n.slice(pt-snr_pre_window, pt).data)
        tr_pt = np.std(tr_s.slice(pt, pt+snr_post_window).data)

    elif mode.lower() == 'sqrt':
        tr_noise = np.sqrt(np.square(
            tr_n.slice(pt-snr_pre_window, pt).data).sum())
        tr_pt = np.sqrt(np.square(
            tr_s.slice(pt, pt+snr_post_window).data).sum())

    snr = tr_pt/tr_noise

    return snr

def snr_pt_v2(tr_vertical, tr_horizontal, pt_p, pt_s, mode='std',
            snr_pre_window=5, snr_post_window=5, highpass=None):
    """
    Calculate snr
    tr_vertical: sac trace vertical component
    tr_horizontal: sac trace horizontal component
    pt_p: p phase utcdatetime object
    pt_s: s phase udtdatetime object
    """
    if highpass:
        tr_vertical = tr_vertical.filter(
            'highpass', freq=highpass).\
            taper(max_percentage=0.1, max_length=0.1)
        tr_horizontal = tr_horizontal.filter(
            'highpass', freq=highpass).\
            taper(max_percentage=0.1, max_length=0.1)
    tr_signal_p = tr_vertical.copy().slice( 
        pt_p, pt_p + snr_pre_window )
    tr_signal_s = tr_horizontal.copy().slice( 
        pt_s, pt_s + snr_pre_window ) 
    tr_noise_p = tr_vertical.copy().slice( 
        pt_p - snr_pre_window, pt_p )
    tr_noise_s = tr_horizontal.copy().slice( 
        pt_s-snr_pre_window, pt_s )
  
    if mode.lower() == 'std':
        snr_p = np.std(tr_signal_p.data)/np.std(tr_noise_p.data)
        snr_s = np.std(tr_signal_s.data)/np.std(tr_noise_s.data)

    elif mode.lower() == 'sqrt':
        snr_p = np.sqrt(np.square(tr_signal_p.data).sum())\
            / np.sqrt(np.square(tr_noise_p.data).sum()) 
        snr_s = np.sqrt(np.square(tr_signal_s.data).sum())\
            / np.sqrt(np.square(tr_noise_s.data).sum()) 

    return snr_p, snr_s

def snr_pt_v3(tr_vertical, tr_horizontal, pt_p, pt_s, mode='std',
            snr_pre_window=5, snr_post_window=5, highpass=None):
    """
    Calculate snr
    tr_vertical: sac trace vertical component
    tr_horizontal: sac trace horizontal component
    pt_p: p phase utcdatetime object
    pt_s: s phase udtdatetime object
    """
    if highpass:
        tr_vertical = tr_vertical.filter(
            'highpass', freq=highpass).taper(
                max_percentage=0.1, max_length=0.1)
        tr_horizontal = tr_horizontal.filter(
            'highpass', freq=highpass).taper(
                max_percentage=0.1, max_length=0.1)
    tr_signal_p = tr_vertical.copy().slice( 
        pt_p, pt_p + snr_pre_window )
    tr_signal_s = tr_horizontal.copy().slice( 
        pt_s, pt_s + snr_pre_window ) 
    tr_noise_p = tr_vertical.copy().slice( 
        pt_p - snr_pre_window, pt_p )
    tr_noise_s = tr_horizontal.copy().slice( 
        pt_s-snr_pre_window, pt_s )
  
    if mode.lower() == 'std':
        snr_p = np.std(tr_signal_p.data)/np.std(tr_noise_p.data)
        snr_s = np.std(tr_signal_s.data)/np.std(tr_noise_s.data)

    elif mode.lower() == 'sqrt':
        snr_p = np.sqrt(np.square(tr_signal_p.data).sum())\
             / np.sqrt(np.square(tr_noise_p.data).sum()) 
        snr_s = np.sqrt(np.square(tr_signal_s.data).sum())\
             / np.sqrt(np.square(tr_noise_s.data).sum()) 

    return snr_p, snr_s

def pick_peaks(prediction, labeled_phase, sac_dt=None,
                     search_win=1, peak_value_min=0.01):
    '''
    search for potential pick
    
    parameters
    ----
    prediction: predicted functions
    labeled_phase: the timing of labeled phase
    sac_dt: delta of sac 
    search_win: time window (sec) for searching 
    local maximum near labeled phases 
    '''
    try:
        tphase = int(round(labeled_phase/sac_dt))
        search_range = [tphase-int(search_win/sac_dt), 
                        tphase+int(search_win/sac_dt)]
        peaks, values = find_peaks(prediction, height=peak_value_min)

        in_search = [np.logical_and(v>search_range[0], 
                        v<search_range[1]) for v in peaks]
        _peaks = peaks[in_search]
        _values = values['peak_heights'][in_search]
        return _peaks[np.argmax(_values)]*sac_dt, \
                _values[np.argmax(_values)]
    except ValueError:
        return -999, -999

def stream_standardize(st):
    '''
    input: obspy.stream object (raw data)
    output: obspy.stream object (standardized)
    '''
    st = st.detrend('demean')
    for s in st:
        s.data /= np.std(s.data)
    return st

def conti_standard_wf(wf, pred_npts, pred_interval_sec, highpass=None):
    '''
    input: 
    wf: obspy.stream object (raw_data)
    pred_npts
    pred_interval_sec
    
    output:
    wf_slices
    wf_start_utc
    '''
    dt = wf[0].stats.sac.delta
    pred_rate = int(pred_interval_sec/dt)
    wf_start_utc = wf[0].stats.starttime
    n_marching_win = int((wf[0].stats.npts - pred_npts)/pred_rate)
    
    wf_slices = []
    for i in range(n_marching_win):
        wf_cp = wf.copy()
        st_idx = pred_rate*i
        end_idx = st_idx + pred_npts

        if highpass:
            wf_cp = wf_cp.detrend('demean')
            wf_cp.filter('highpass', freq=highpass)

        # data standarization
        wf_E = wf_cp[0].data[st_idx:end_idx]
        wf_E-=np.mean(wf_E); wf_E/=np.std(wf_E)

        wf_N = wf_cp[1].data[st_idx:end_idx]
        wf_N-=np.mean(wf_N); wf_N/=np.std(wf_N)

        wf_Z = wf_cp[2].data[st_idx:end_idx]
        wf_Z-=np.mean(wf_Z); wf_Z/=np.std(wf_Z)
        
        wfs = np.array([wf_E, wf_N, wf_Z]).T
        wf_slices.append(wfs)
    return np.array(wf_slices), wf_start_utc, dt

if __name__ == '__main__':
    pass
