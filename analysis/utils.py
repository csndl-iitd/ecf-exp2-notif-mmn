import numpy as np
import mne

# config file

markers = { 
        'B1_Boundary': [31],
        'B1_baseline_start': [32], 
        'B1_baseline_end': [33],
        'CN_standard_tone_start' : [34],
        'CN_deviant_tone_start' :[35],
        #'B1_standard_tone_end' : [16],
        #'B1_deviant_tone_end' :[18],
        'B2_Boundary': [21],
        'B2_baseline_start': [22],
        'B2_baseline_end': [23],
        'PN_standard_tone_start' : [24],
        'BT_deviant_tone_start' :[25],
        #'B2_standard_tone_end' : [26],
        #'B2_deviant_tone_end' :[28],
        'B3_Boundary': [11],
        'B3_baseline_start': [12],
        'B3_baseline_end': [13],
        'BT_standard_tone_start' : [14],
        'PN_deviant_tone_start' :[15],
    }


config = {
     "data_folder" :  r"D:\Work_data\Notification_evoked_response\Data_at_IITD/",
     "data_type" : 'EEG',
     'resample' : 512,
     "ref_electrode" :['TP9', 'TP10'] ,   #"list of electroder or rerefeence method",
     'filter' :  [.1,30] ,                #'list with low and high frequency',
     'eog_chn' : ['Fp1', 'Fp2',"AF7","AF8",'F7', 'F8'],           #  'list of  channels to detect eog', 
     'event_IDs' : [v[0] for k, v in markers.items() if 'standard_tone_start' in k or 'deviant_tone_start' in k] ,#"list of event ids for epoching"
     'epoch_dur' : [-0.300,0.700 ], #'list of inital and end time',
     'num_ICA' :  20 ,  #"Number of ica component",
     'CN_devi_erp_evoked_dict': {},
     'BT_devi_eps_evoked_dict' :{},
     'PN_devi_eps_evoked_dict' : {},
     'CN_sta_erp_evoked_dict' : {},
     'BT_sta_eps_evoked_dict' : {},
     'PN_sta_eps_evoked_dict'  : {},
     }


## Function to get check event in the epochs
def available_event_(Events,evn = config['event_IDs']):
            unique_event_ids = np.unique(Events[:, 2])
            non_event = []
            event_IDs = []
            for event_id in evn:
                if event_id not in unique_event_ids:
                    non_event.append(event_id)
                else:
                    event_IDs.append(event_id)
            if non_event:
                    print(non_event, " are/ is not available events IDs.")
                    print(event_IDs, " are/ is available events IDs." )
            else:
                    print(event_IDs, " are/ is available events IDs." )
            return event_IDs 
## Some of the common function 
def identify_bad_channels(data, threshold=80e-6):
    """
    Identify bad channels based on amplitude difference exceeding the threshold.
    
    :param data: 3D array of shape (n_epochs, n_channels, n_times)
    :param threshold: Amplitude difference threshold in Volts (default: 80 microvolts)
    :return: List of bad channel indices
    """
    amplitude_diff = np.ptp(data, axis=2)  # peak-to-peak amplitude for each epoch and channel
    bad_channels = np.where(np.any(amplitude_diff > threshold, axis=0))[0]
    return bad_channels.tolist()


def remove_epochs_with_bad_mmn_channels(epochs, mmn_channels, amplitude_threshold=80e-6):
    """
    Remove epochs where MMN-related channels exceed the amplitude threshold.
    
    :param epochs: mne.Epochs object
    :param mmn_channels: List of MMN channel names
    :param amplitude_threshold: Amplitude difference threshold in Volts (default: 80 microvolts)
    :return: mne.Epochs object with bad epochs removed
    """
    data = epochs.get_data()
    
    
    bad_mmn_epochs = []
    for epoch_idx in range(len(epochs)):
        epoch_data = data[epoch_idx]
        epoch_bad_channels = identify_bad_channels(epoch_data[np.newaxis, :, :], amplitude_threshold)
        
        for ch_idx in epoch_bad_channels:
            ch_name = epochs.ch_names[ch_idx]
            if any(mmn_ch in ch_name for mmn_ch in mmn_channels):
                bad_mmn_epochs.append(epoch_idx)
                break
    
    good_epochs = np.setdiff1d(np.arange(len(epochs)), bad_mmn_epochs)
    return epochs[good_epochs]



def Average_in_trials_in_time_window(Epoches,chan, stime,etime):
    # Crop the epochs to the desired time window (100 ms to 500 ms)
    cropped_epochs =Epoches.copy().pick(chan).crop(tmin=stime, tmax=etime)
    # Initialize a dictionary to store averages for each event\]
    event_averages = {}
    # Iterate over event types
    for event_id in cropped_epochs.event_id:
        # Extract epochs for the current event
        event_epochs = cropped_epochs[event_id]
        # Calculate average for each trial across time points
        trial_averages = np.mean(event_epochs.get_data(), axis=2)  # Average across time (axis=2)
        # Store in the dictionary
        event_averages[event_id] = trial_averages
    # Example: Access the average data for a specific event
    for event, data in event_averages.items():
        print(f"Event: {event}, Averages shape: {data.shape}")
    return event_averages


def remove_epochs_with_n_bad_channels(epochs, bad_chn_thresold, amplitude_threshold=80e-6):
    """
    Remove epochs if n number of bad channel by thresold remove it
    
    :param epochs: mne.Epochs object
    :param bad_chn_thresold : check n number of bad channel
    :param amplitude_threshold: Amplitude difference threshold in Volts (default: 80 microvolts)
    :return: mne.Epochs object with bad epochs removed
    """
    data = epochs.get_data()
    
    
    bad_mmn_epochs = []
    for epoch_idx in range(len(epochs)):
        epoch_data = data[epoch_idx]
        epoch_bad_channels = identify_bad_channels(epoch_data[np.newaxis, :, :], amplitude_threshold)
        
        if len(epoch_bad_channels) > bad_chn_thresold:
                bad_mmn_epochs.append(epoch_idx)
                break
        
    good_epochs = np.setdiff1d(np.arange(len(epochs)), bad_mmn_epochs)
    return epochs[good_epochs]



def remove_or_interpolate_epochs_with_n_bad_chn(epochs, bad_chn_threshold, amplitude_threshold=80e-6):
    """
    Remove epochs if the number of bad channels exceeds a threshold, otherwise interpolate the bad channels.

    :param epochs: mne.Epochs object
    :param bad_chn_threshold: Maximum number of bad channels allowed before rejecting the epoch
    :param amplitude_threshold: Amplitude threshold in Volts to mark a channel as bad (default: 80 microvolts)
    :return: New mne.Epochs object with bad epochs removed or interpolated
    """
    data = epochs.get_data()
    good_epochs = []

    for epoch_idx in range(len(epochs)):
        epoch_data = data[epoch_idx]
        bad_channel_indices = identify_bad_channels(epoch_data[np.newaxis, :, :], amplitude_threshold)
        bad_channel_names = [epochs.ch_names[i] for i in bad_channel_indices]

        if len(bad_channel_indices) > bad_chn_threshold:
            continue  # Skip this epoch
        else:
            # Create a copy of the single epoch
            epoch = epochs[epoch_idx].copy()
            if len(bad_channel_indices) > 0:
                epoch.info['bads'] = bad_channel_names
                epoch.interpolate_bads(reset_bads=True)
            good_epochs.append(epoch)

    # Concatenate the list of good/interpolated single-epoch Epochs into a new Epochs object
    if good_epochs:
        return mne.concatenate_epochs(good_epochs)
    else:
        return None  # or raise an error if all epochs are bad
