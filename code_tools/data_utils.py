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


if __name__ == '__main__':
    pass
