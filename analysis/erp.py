# read the epoch for each subject and return the ERPs
import mne
import sys
import os
sys.path.append(os.path.abspath('..'))
from utils import remove_epochs_with_n_bad_channels, config ,available_event_,markers 

def event_ERP(sub_id, cleaned_epochs,config):
    #
    event_ls = available_event_(cleaned_epochs.events,config['event_IDs'])
    for event in event_ls:
        if event  ==  markers['BT_standard_tone_start'][0]:
            config['BT_sta_eps_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()
        elif event  ==  markers['BT_deviant_tone_start'][0]:
            config['BT_devi_eps_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()
        elif event  ==  markers['CN_standard_tone_start'][0]:
            config['CN_sta_erp_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()
        elif event  ==  markers['CN_deviant_tone_start'][0]:
            config['CN_devi_erp_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()
        elif event  ==  markers['PN_standard_tone_start'][0]:
            config['PN_sta_eps_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()
        elif event  ==  markers['BT_standard_tone_start'][0]:
            config['BT_sta_eps_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()
        elif event  ==  markers['PN_deviant_tone_start'][0]:
            config['PN_devi_eps_evoked_dict'][sub_id] =cleaned_epochs[str(event)].average()      
    del cleaned_epochs
