import os
from os import path

import numpy as np
import pandas as pd

from .defs import *

def epochs_to_df_with_annotations(epochs):
    """Convert MNE Epochs to a DataFrame with annotations as multi-index.
    Annotations include epoch number, condition, validity and index from last deviant.

    Args:
        epochs (mne.Epochs): MNE Epochs object.
    Returns:
        pd.DataFrame: DataFrame with multi-index (subject, condition, trial, time).
    """
    # extract annotations and identify bad epochs
    annots = epochs.annotations.to_data_frame()
    # remove duplicated rows with the same onset time
    annots['dup'] = annots.onset.astype(int).diff()!=0
    annots = annots[annots.dup].drop('dup', axis=1)
    # identify bad epochs based on 'bad_' annotations
    # this is done by marking bad epochs as those whose onset time is between
    # the onset and duration of a 'bad_' annotation
    annots['time'] = annots['onset'].astype(int)
    annots.loc[annots['description'].str.contains('bad'), 'bad'] = annots['onset'].astype(int)+annots['duration']*1e9
    annots = annots.ffill()
    annots['dropped'] = annots['time']<=annots['bad']
    # remove all rows with 'bad' in description
    annots = annots[~annots['description'].str.contains('bad')]
    annots['description'] = annots.description.map(lambda s: s.split('/')[1].replace(' ', ''))
    annots_epoch = pd.Series(np.nan, index=annots.index)
    annots_epoch[annots.description.isin(triggers.values())] = 1
    annots['epoch'] = annots_epoch.cumsum() - 1
    annots = annots.dropna(subset=['epoch']).set_index(['epoch'])

    epochs = epochs.to_data_frame().set_index(
        ['epoch', 'condition', 'time']
    )[locations['fronto_central_midline']].mean(axis=1).unstack('time')
    _epoch_meta = epochs.index.to_frame().reset_index(drop=True)
    _epoch_meta['valid'] = ~annots['dropped']
    _epoch_meta['condition'] = _epoch_meta['condition'].map(lambda s: int(s[1:]))
    _epoch_meta['cblock'] = (_epoch_meta['condition'].diff().fillna(0)>0).cumsum()
    _epoch_meta['cindex'] = _epoch_meta.groupby('cblock').apply(
        lambda df: pd.Series(range(len(df)), index=df.index), include_groups=False
    ).droplevel(0)
    epochs.index = pd.MultiIndex.from_frame(_epoch_meta)
    return epochs

def read_n1_scores(dir, stim_type='STD', electrodes='fronto_central_midline'):
    """Read precomputed N1 scores from CSV file.

    Args:
        dir (str): Directory where the scores CSV file is located.
        stim_type (str): Type of stimulus ('STD', 'DEV', etc.).
        electrodes (str): Electrode configuration used.
    Returns:
        pd.DataFrame: DataFrame containing N1 scores.
    """
    scores_file = path.join(dir, f'scores_n1_{stim_type}_{electrodes}.csv')
    if not path.exists(scores_file):
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    
    n1_scores = pd.read_csv(scores_file, index_col=[0, 1, 2])
    return n1_scores

def aggregate_n1_scores(n1_scores):
    """Aggregate N1 scores by stimulus and subject.

    Args:
        n1_scores (pd.DataFrame): DataFrame containing N1 scores.
    Returns:
        pd.DataFrame: Aggregated N1 scores with mean and std.
    """
    n1_agg = n1_scores.groupby(['stim', 'subject']).agg(['mean', 'std'])
    n1_agg.columns = ['_'.join(c) for c in n1_agg.columns]
    return n1_agg

def read_n1_erps(dir, stim_type='STD', electrodes='fronto_central_midline'):
    """Read precomputed N1 ERPs from CSV file.

    Args:
        dir (str): Directory where the ERPs CSV file is located.
        stim_type (str): Type of stimulus ('STD', 'DEV', etc.).
        electrodes (str): Electrode configuration used.
    Returns:
        pd.DataFrame: DataFrame containing N1 ERPs.
    """
    erps_file = path.join(dir, f'erps_n1_{stim_type}_{electrodes}.csv')
    if not path.exists(erps_file):
        raise FileNotFoundError(f"ERPs file not found: {erps_file}")
    
    n1_erps = pd.read_csv(erps_file, header=[0, 1], index_col=0)
    return n1_erps

def read_mmn_scores(dir, mmn_type='across', electrodes='fronto_central_midline'):
    """Read precomputed MMN scores from CSV files.

    Args:
        dir (str): Directory where the scores CSV files are located.
        mmn_type (list): List of stimulus types to read.
        electrodes (str): Electrodes included in the analysis.
    Returns:
        dict: Dictionary with stimulus types as keys and DataFrames as values.
    """
    scores_file = path.join(dir, f'scores_mmn_{mmn_type}_{electrodes}.csv')
    if not path.exists(scores_file):
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    
    mmn_scores = pd.read_csv(scores_file, index_col=[0, 1, 2])
    return mmn_scores

def aggregate_mmn_scores(mmn_scores):
    """Aggregate MMN scores by stimulus and subject.

    Args:
        mmn_scores (dict): Dictionary with stimulus types as keys and DataFrames as values.
    Returns:
        dict: Dictionary with stimulus types as keys and aggregated DataFrames as values.
    """
    mmn_agg = mmn_scores.groupby(['stim', 'subject']).agg(['mean', 'std'])
    mmn_agg.columns = ['_'.join(c) for c in mmn_agg.columns]
    return mmn_agg

def read_mmn_erps(dir, mmn_type='across', electrodes='fronto_central_midline'):
    """Read precomputed MMN ERPs from CSV files.

    Args:
        dir (str): Directory where the ERPs CSV files are located.
        stim_types (list): List of stimulus types to read.
    Returns:
        dict: Dictionary with stimulus types as keys and DataFrames as values.
    """
    erps_file = path.join(dir, f'erps_mmn_{mmn_type}_{electrodes}.csv')
    if not path.exists(erps_file):
        raise FileNotFoundError(f"ERPs file not found: {erps_file}")
    
    mmn_erps = pd.read_csv(erps_file, header=[0, 1], index_col=0)
    return mmn_erps