import numpy as np
import scipy.signal as sig
import scipy.stats as st
from itertools import combinations


def fcd_ccor_eeg(ts, win_len=30, win_sp=1):
    n_nodes, n_samples = ts.shape
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    hillbert = sig.hilbert(ts, axis=0)
    phases = np.unwrap(np.angle(hillbert))
    mean_phases =np.apply_along_axis(np.mean, 1, phases)
    sines = np.sin(np.subtract(phases, mean_phases.reshape((n_nodes, 1))))
    
    fc_stack = np.zeros((int((n_samples-win_len)/win_sp)+1, len(fc_triu_ids[0])))

    for i, t0 in enumerate(range( 0, n_samples-win_len, win_sp )):
        fc = np.zeros((n_nodes, n_nodes))
        t1=t0+win_len
        for (j, k) in combinations(range(n_nodes), r=2):
            fc[j,k] = np.sum(sines[j,t0:t1]*sines[k,t0:t1])/np.sqrt(np.sum(sines[j,t0:t1]**2*sines[k,t0:t1]**2))
        fc_stack[i] = fc[fc_triu_ids]
        
    FCD = np.corrcoef(fc_stack)
    return FCD, fc_stack


def fcd_sim(ts, win_len=30, win_sp=1):
    """
    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
    """
    n_samples, n_nodes = ts.shape
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)

    fcs = np.array(fc_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs


def fluidity(fcd, win_len, overlap):
    triangle = np.triu(fcd, (float(win_len)/(float(win_len)-float(overlap)))+1)
    return np.var(triangle)


def gap(ts):
    z_scored = st.zscore(ts, axis=1)
    return np.gradient(np.sqrt(np.sum(z_scored**2, axis=0))).max()


def n_states_eeg(avalanches):
    avals = []
    Zbin = avalanches['Zbin'].T
    for j,k in avalanches['ranges']:
        if Zbin[:,j:k].shape[1] < 3:
            continue
        else:
            aval = np.where(np.any(Zbin[:,j:k].squeeze(), axis=1), 1,0)
            avals.append(aval)    
    return np.unique(np.stack(avals), axis=0).shape[0]