from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne

def compute_single_erp_scores(epochs, window, latency_fractions=[0.25], picks=None, extrema=-1):
    '''
    Returns the following scores for an ERP waveform derived by averaging the epochs
    1. peak amplitude (of the highest peak in the window) [PA]
    2. peak latency (of the highest peak in the window) [PL]
    3. time-window mean amplitude [TWMA]
    4. fractional area latency [FAL]
    '''
    if picks is None:
        picks = epochs[0].columns
    erp = epochs[0][picks].mean(axis=1).groupby('time').mean()
    erp_win = erp.loc[window[0]:window[1]]
    # find the peak of the erp and its amplitude
    t = (extrema*erp_win).idxmax()
    a = erp.loc[t]
    # find the integrated area in the window for the erp
    ia = erp_win.mean()
    # find the latencies for different fractional coverages
    tfs = {}
    # TODO: fractional area latency needs to be fixed!
    for f in latency_fractions:
        tfs[f'FAL_{f}'] = erp_win.index[extrema*erp_win.cumsum()>extrema*erp_win.sum()*f][0]
    return pd.concat([pd.Series(dict(PA=a, PL=t, TWMA=ia)), pd.Series(tfs)], names=['iteration']), [erp]

def compute_dwave_scores(epochs, window, latency_fractions=[0.25], picks=None, extrema=-1):
    '''
    Returns the following scores for an ERP difference wave derived by averaging
    the two groups of epochs and subtracting group 1 from group 0
    1. peak amplitude (of the highest peak in the window) [PA]
    2. peak latency (of the highest peak in the window) [PL]
    3. time-window mean amplitude [TWMA]
    4. fractional area latency [FAL]
    '''
    if picks is None:
        picks = epochs[0].columns
    erp0 = epochs[0][picks].mean(axis=1).groupby('time').mean()
    erp1 = epochs[1][picks].mean(axis=1).groupby('time').mean()
    dwave = erp0 - erp1
    erp_win = dwave.loc[window[0]:window[1]]
    # find the peak of the erp and its amplitude
    t = (extrema*erp_win).idxmax()
    a = dwave.loc[t]
    # find the integrated area in the window for the erp
    ia = erp_win.mean()
    # find the latencies for different fractional coverages
    tfs = {}
    # TODO: fractional area latency needs to be fixed!
    for f in latency_fractions:
        tfs[f'FAL_{f}'] = erp_win.index[extrema*erp_win.cumsum()>extrema*erp_win.sum()*f][0]
    return pd.concat([pd.Series(dict(PA=a, PL=t, TWMA=ia)), pd.Series(tfs)], names=['iteration']), [dwave]

def compute_bootstrapped_scores(epochs, fn, n_iterations, sample_size=None, **kwargs):
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
    # convert epochs to dataframes for faster computation inside the loop
    epochs = [
        e.to_data_frame().drop('condition', axis=1).set_index(['time', 'epoch']).unstack('time') for e in epochs
    ]
    scores, erps = {}, {}
    itr, errs = 0, []
    with tqdm(total=n_iterations, desc='iter') as pbar:
        while itr < n_iterations:
            try:
                _epochs = []
                # generate bootstrapped epochs
                for epoch, ss in zip(epochs, sample_size):
                    _epochs.append(
                        epoch.sample(ss, replace=True).stack('time', future_stack=True)
                    )
                # generate corresponding scores
                scores[itr], erps[itr] = fn(_epochs, **kwargs)
                itr += 1
                pbar.update(1)
            except Exception as e:
                errs.append(e)
                # print(e)
                # if too many iterations are failing, break to get out of the loop
                if len(errs)>=100:
                    break
    return pd.concat(scores, names=['iteration', 'score']).unstack('score'), erps, errs

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
        scores, erps, errs = compute_bootstrapped_scores(
            epochs, compute_single_erp_scores, n_iterations=n_iterations, sample_size=sample_size, **kwargs
        )
    elif len(epochs)==2:
        scores, erps, errs = compute_bootstrapped_scores(
            epochs, compute_dwave_scores, n_iterations=n_iterations, sample_size=sample_size, **kwargs
        )
    else:
        raise ValueError('exactly one or two sets of epochs must be given.')
    erps = pd.concat({k:v[0] for k, v in erps.items()}, axis=1)

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
    return scores, erps, errs, f