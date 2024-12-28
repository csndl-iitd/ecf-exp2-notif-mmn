import numpy as np


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