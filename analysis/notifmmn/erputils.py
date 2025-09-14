from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne

def _compute_peak_latency_amplitudes(erps, window, extrema=-1):
    '''
    Computes the peak latency and peak amplitude of a positive or negative peak
    (determined by extrema parameter) within a given window.
    '''
    pls = (extrema*erps.loc[window[0]:window[1]]).idxmax()
    pas = pd.Series([erps[c].loc[t] for c,t in zip(erps.columns, pls)], index=pls.index)
    # peaks at boundaries should not be considered peaks, so we require
    # that peaks are at least 5 ms away fron boundaries of the window
    pls[(pls<window[0]+0.005)|(pls>window[1]-0.005)] = np.nan
    pas[(pls<window[0]+0.005)|(pls>window[1]-0.005)] = np.nan
    return pd.concat(dict(
        PL=pls,
        PA=pas
    ), axis=1)

def _compute_areal_latency_amplitude(erps, window, latency_fractions=[0.25], extrema=-1):
    '''
    Computes the area under curve within the window to obtain the amplitude
    and the times at which the area crosses certain fractions to obtain latency.
    '''
    erps = (erps*extrema).loc[window[0]:window[1]]
    # detrend the ERPs so that end points are both at 0
    erps = erps.apply(lambda c: c - c.iloc[0] - (c.iloc[-1]-c.iloc[0])*np.arange(len(c))/len(c))
    amp = erps.sum()
    cs = erps.cumsum()
    tfs = {'TWMA':amp/len(erps)*extrema}
    for f in latency_fractions:
        tfs[f'FAL_{f}'] = (cs < amp*f).cumsum().idxmax()
    return pd.concat(tfs, axis=1)

def compute_single_erp_scores(erps, window, latency_fractions=[0.25], extrema=-1):
    '''
    Returns the following scores for an ERP waveform derived by averaging the epochs
    1. peak amplitude (of the highest peak in the window) [PA]
    2. peak latency (of the highest peak in the window) [PL]
    3. time-window mean amplitude [TWMA]
    4. fractional area latency [FAL]
    erps is a list of pandas dataframes
    '''
    return pd.concat([
        _compute_peak_latency_amplitudes(erps[0], window, extrema),
        _compute_areal_latency_amplitude(erps[0], window, latency_fractions, extrema)
    ], axis=1)

def compute_dwave_scores(erps, window, latency_fractions=[0.25], picks=None, extrema=-1):
    '''
    Returns the following scores for an ERP difference wave derived by averaging
    the two groups of epochs and subtracting group 1 from group 0
    1. peak amplitude (of the highest peak in the window) [PA]
    2. peak latency (of the highest peak in the window) [PL]
    3. time-window mean amplitude [TWMA]
    4. fractional area latency [FAL]
    '''
    dwave = erps[0] - erps[1]
    return pd.concat([
        _compute_peak_latency_amplitudes(dwave, window, extrema),
        _compute_areal_latency_amplitude(dwave, window, latency_fractions, extrema)
    ], axis=1)

def compute_bootstrapped_scores(epochs, fn, n_iterations, sample_size=None, picks=None, **kwargs):
    '''
    Compute score(s) with SEM by bootstrapping over epochs
    epochs: an instance of epochs, or list of epoch instances when more than one are required (for example for a difference wave)
    fn: the function to be called to evaluate the parameters for each resampling of epochs
    fn must accept a list of epochs as a pandas dataframe (indexed by epoch and time)
    fn must return a set of scores as a pandas series and the list of generated erps for each input epoch
    n_iterations: number of times resampling should be performed
    sample_size: int or list of ints (number of samples to draw in each iteration). default None will use the same number of samples as original epochs instance
    kwargs: arguments to be passed to the function fn
    '''
    if not isinstance(epochs, list):
        epochs = [epochs]
    if sample_size is None:
        sample_size = [len(x) for x in epochs]
    if picks is None:
        picks = epochs[0].columns
    
    # convert epochs to dataframes for faster computation inside the loop
    epochs = [
        e.to_data_frame().drop('condition', axis=1).set_index(['time', 'epoch'])[picks]\
            .unstack('time').T.groupby('time').mean().T for e in epochs
    ]
    erp_list = []
    for e, ss in zip(epochs, sample_size):
        erps = {}
        for itr in range(n_iterations):
            erps[itr] = e.sample(ss, replace=True).mean()
        erp_list.append(pd.concat(erps, axis=1))
    return fn(erp_list, **kwargs), erp_list

def get_erp_sem(epochs, n_iterations, sample_size=None, verifiplot=True, **kwargs):
    '''
    Returns the scores, erps, errors (encountered in each bootstrapped iteration) and figure object (if verifiplot)
    epochs: if single epochs instance, computes single erp scores
            if list of two epochs, computes the scores for the difference wave
    n_iterations: number of bootstrap iterations to run
    sample_size: number of epochs in each bootstrap iteration (defaults to the number of epochs in original epochs)
    verifiplot: setting True (default) will plot the ERPs (or difference waves) across bootstrapped iterations
    kwargs: arguments required for single wave or difference wave score computation
    kwargs include window (tuple), latency_fractions (list, for computing fractional area latencies), picks (to average over), extrema (whether to find minima or maxima)
    '''
    if not isinstance(epochs, list):
        epochs = [epochs]
    if len(epochs)==1:
        scores, erps = compute_bootstrapped_scores(
            epochs, compute_single_erp_scores, n_iterations=n_iterations, sample_size=sample_size, **kwargs
        )
        erps = erps[0]
    elif len(epochs)==2:
        scores, erps = compute_bootstrapped_scores(
            epochs, compute_dwave_scores, n_iterations=n_iterations, sample_size=sample_size, **kwargs
        )
        erps = erps[0] - erps[1]
    else:
        raise ValueError('exactly one or two sets of epochs must be given.')

    f = None
    if verifiplot:
        window = kwargs.pop('window')
        f, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
        erps.plot(ax=ax, c='C0', alpha=0.01, legend=False)
        ax.scatter(scores.PL, scores.PA, s=1, c='k')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('voltage ($\mu V$)')
        ax.axvspan(*window, color='gray', alpha=0.1)
        ax.annotate(
            scores.aggregate(['mean', 'std']).to_string(float_format='{:.3f}'.format, col_space=8, justify='right'),
            (10, 10), xycoords='axes pixels', family='monospace'
        )
    return scores, erps, f