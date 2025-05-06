#Read a raw EEG data and return preprocessed Epoched data for that subject
# read the data

import numpy as np

import pandas as pd
import mne
import os
from mne.preprocessing import ICA

import sys
sys.path.append(os.path.abspath('..'))
from utils import remove_epochs_with_n_bad_channels, config,available_event_


def EEG_preprocessing(sub_eeg_path, config):
    """     Input : single subject EEG data path with config
            output: Cleaned preprocessed epoch
    """
    for eeg in os.listdir(sub_eeg_path):
        if eeg[-5:] == ".vhdr":
            raw = mne.io.read_raw_brainvision(os.path.join(sub_eeg_path,eeg), preload=True)
            raw = mne.add_reference_channels(raw, ref_channels=["Cz"])
            raw.set_montage("easycap-M1")
            raw.set_eeg_reference(ref_channels=config['ref_electrode'])
            raw.resample(config['resample'])
            # filter 
            raw.filter(l_freq=config['filter'][0],h_freq=config['filter'][1]  )
            print(config['event_IDs'])
            #ICA 
            ica = ICA(n_components=config['num_ICA'], random_state=97)
            ica.fit(raw)
            print("ICA Completed ...")
            
            #removing Eye movement artifacts 
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=config['eog_chn'])
            print(eog_indices, "EoG event ICA component ")
            ica.exclude = eog_indices
            ica.apply(raw)
            events,event_dict = mne.events_from_annotations(raw)
            
            event_ids = available_event_(events, evn = config['event_IDs'])
            # Epoch segments for ERP
            epochs = mne.Epochs(raw, events, event_id=event_ids,baseline=(None, 0), tmin=config['epoch_dur'][0], tmax=config['epoch_dur'][1], preload=True)
            # Remove Epoch or interpolate the bad channels 
            cleaned_epochs = remove_epochs_with_n_bad_channels(epochs, bad_chn_thresold = 8, amplitude_threshold=80e-6)
            try:
                cleaned_epochs.drop_channels(['GSR', 'PPG'])
            except:
                print("NO GSR & PPG sensor")
                pass


            print(f"Original number of epochs: {len(epochs)}")
            print(f"Number of epochs after removal: {len(cleaned_epochs)}")
 
            """if len(cleaned_epochs['14','15']) < 450 or  len(cleaned_epochs['24','25']) < 450 :
                epoch_per_num["subject Dropped"].append(sub) 
                print("B1 len :",len(cleaned_epochs['14','15']),"B2 len :",len(cleaned_epochs['24','25']))
                continue"""
            return cleaned_epochs