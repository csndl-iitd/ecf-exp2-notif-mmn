import matplotlib.pyplot as plt

from mne import EvokedArray
import numpy as np

def get_fractional_area_latency(waveform, times, fraction=0.50):
    """Determine the latency at which a fraction of the area under the curve is reached."""
    # For MMN, we're interested in the negative area
    neg_waveform = np.minimum(waveform, 0)  # Keep only negative values, set positive to 0
    
    # Calculate cumulative area
    area = np.trapz(neg_waveform, times)
    if area >= 0:  # If no negative area (no clear MMN)
        return np.nan
        
    cumulative_area = np.zeros_like(times)
    for i in range(1, len(times)):
        cumulative_area[i] = np.trapz(neg_waveform[:i+1], times[:i+1])
    
    # Find the timepoint where the cumulative area reaches the target fraction
    threshold = fraction * area
    crossing_idx = np.argmin(np.abs(cumulative_area - threshold))
    
    return times[crossing_idx]

def get_onset_latency(waveform, times, percent_threshold=0.50):
    """Determine the onset latency when signal reaches percent_threshold of peak amplitude."""
    peak_idx = np.argmin(waveform)
    peak_amplitude = waveform[peak_idx]
    
    # Threshold is a percentage of peak amplitude
    threshold = peak_amplitude * percent_threshold
    
    # Find the first point where amplitude exceeds threshold
    for i in range(peak_idx):
        if waveform[i] <= threshold:  # For negative MMN, we use <= 
            return times[i]
    
    # If no crossing found, return NaN
    return np.nan

def extract_mmn_latencies(case1_dict, case2_dict, time_window=(100, 300), channels=None, 
                          method='peak', fractional_area=0.50):
    """Extract MMN latencies using specified method."""
    common_subjects = sorted(set(case1_dict.keys()) & set(case2_dict.keys()))
    time_axis = None  # Will be determined from the first subject's data
    
    latencies_case1 = {}
    latencies_case2 = {}
    
    for subject in common_subjects:
        # Get the MMN evoked data for selected channels
        if channels:
            case1_data = case1_dict[subject].copy().pick(channels)
            case2_data = case2_dict[subject].copy().pick(channels)
        else:
            case1_data = case1_dict[subject].copy()
            case2_data = case2_dict[subject].copy()
        
        # Extract the time axis if not yet done
        if time_axis is None:
            time_axis = case1_data.times
            
        # Crop to time window of interest
        case1_data = case1_data.copy().crop(tmin=time_window[0]/1000, tmax=time_window[1]/1000)
        case2_data = case2_dict[subject].copy().crop(tmin=time_window[0]/1000, tmax=time_window[1]/1000)
        
        # Get the data as numpy arrays - average across selected channels
        case1_avg = np.mean(case1_data.data, axis=0) if len(case1_data.data.shape) > 1 else case1_data.data
        case2_avg = np.mean(case2_data.data, axis=0) if len(case2_data.data.shape) > 1 else case2_data.data
        
        # Get the time axis for this window
        window_times = case1_data.times
        
        # Extract latency based on the selected method
        if method == 'peak':
            # Find the peak latency (time of max negative amplitude)
            peak_idx1 = np.argmin(case1_avg)
            peak_idx2 = np.argmin(case2_avg)
            lat1 = window_times[peak_idx1] * 1000  # Convert to ms
            lat2 = window_times[peak_idx2] * 1000
            
        elif method == 'fractional_area':
            # Fractional area latency (more robust against noise)
            lat1 = get_fractional_area_latency(case1_avg, window_times, fractional_area) * 1000
            lat2 = get_fractional_area_latency(case2_avg, window_times, fractional_area) * 1000
            
        elif method == 'onset':
            # Onset latency (when amplitude reaches a percentage of peak)
            percent_threshold = 0.50  # Typically 50% of peak amplitude
            lat1 = get_onset_latency(case1_avg, window_times, percent_threshold) * 1000
            lat2 = get_onset_latency(case2_avg, window_times, percent_threshold) * 1000
            
        latencies_case1[subject] = lat1
        latencies_case2[subject] = lat2
    
    return latencies_case1, latencies_case2, common_subjects



def align_and_standardize_evokeds(evoked_dict, threshold_results, condition_key):
    """
    Align evoked responses to threshold points and standardize time ranges.
    
    Parameters:
    -----------
    evoked_dict : dict
        Dictionary with subject IDs as keys and evoked objects as values
    threshold_results : dict
        Pre-calculated results from process_all_subjects
    condition_key : str
        Key to access condition-specific threshold times ('beep' or 'sn')
    tmin, tmax : float
        Standardized time range in seconds, relative to threshold point
        
    Returns:
    --------
    dict
        Dictionary with subject IDs as keys and standardized evoked objects
    """
    aligned_dict = {}
    
    # Use only subjects that exist in both dictionaries and have valid thresholds
    common_subjects = set(evoked_dict.keys()) & set(threshold_results.keys())
    
    valid_subjects = []
    
    for subject_id in common_subjects:
        # Get the pre-calculated threshold time
        threshold_time = threshold_results[subject_id][condition_key]['time']
        
        if threshold_time is not None:
            # Make a copy to avoid modifying the original
            evoked_copy = evoked_dict[subject_id].copy()
            
            # Shift time to center at the threshold point
            evoked_copy.shift_time(-threshold_time)
            
            
            # Store the aligned and cropped evoked object
            aligned_dict[subject_id] = evoked_copy
            valid_subjects.append(subject_id)
    
    print(f"Successfully aligned {len(valid_subjects)} subjects: {valid_subjects}")
    return aligned_dict


def update_evoked_times(aligned_dict, atol: float = 1e-8) :
    """
    Return a new list of EvokedArray objects where:
      - The time point closest to zero is set to 0.0,
      - Time points after it are recomputed using sampling rate,
      - Time points before it are unchanged.
    
    Parameters
    ----------
    evokeds : list of mne.Evoked
        List of Evoked objects to update.
    atol : float
        Tolerance to consider a time point effectively zero.

    Returns
    -------
    updated_evokeds : list of mne.EvokedArray
        List of EvokedArray objects with updated timing.
    """
    evoked_list = list(aligned_dict.values())
    updated_evokeds = []
    
    for evoked in evoked_list:
        times = evoked.times
        sfreq = evoked.info['sfreq']
        dt = 1.0 / sfreq

        zero_idx = np.argmin(np.abs(times))
        if not np.isclose(times[zero_idx], 0.0, atol=atol):
            print(f"Warning: Closest to zero is {times[zero_idx]:.6f}, not within atol={atol}")

        # Calculate new tmin such that time at zero_idx becomes 0
        new_tmin = -zero_idx * dt

        # Create new evoked with same data, info, and updated tmin
        new_evoked = EvokedArray(evoked.data, evoked.info, tmin=new_tmin, nave=evoked.nave,
                                 comment=evoked.comment, kind=evoked.kind)
        updated_evokeds.append(new_evoked)

    return updated_evokeds

def extract_peak_amplitude(evoked_dict, time_window, polarity='negative', channels=None):
    """
    Extract peak amplitude from evoked responses.
    
    Parameters:
    -----------
    evoked_dict : dict
        Dictionary with subject IDs as keys and evoked objects as values
    time_window : tuple
        Time window to search for peak
    polarity : str
        'negative' for MMN, 'positive' for positive components
    channels : list or None
        Specific channels to analyze
        
    Returns:
    --------
    dict
        Dictionary with subject IDs and their peak amplitudes
    """
    amplitudes = {}
    
    for subject_id, evoked in evoked_dict.items():
        # Select channels if specified
        if channels:
            evoked_copy = evoked.copy().pick(channels)
        else:
            evoked_copy = evoked.copy()
        
        # Crop to time window
        evoked_cropped = evoked_copy.crop(tmin=time_window[0], tmax=time_window[1])
        
        # Get data and find peak
        data = np.mean(evoked_cropped.data, axis=0)  # Average across channels
        
        if polarity == 'negative':
            peak_amp = np.min(data)
        else:
            peak_amp = np.max(data)
            
        amplitudes[subject_id] = peak_amp
    
    return amplitudes

def extract_mean_amplitudes(case1_dict, case2_dict, time_window=(100, 250), channels=None):
    """Extract mean MMN amplitudes within the specified time window for each subject."""
    # Ensure both dictionaries have the same subjects
    common_subjects = sorted(set(case1_dict.keys()) & set(case2_dict.keys()))
    
    amplitudes_case1 = []
    amplitudes_case2 = []
    
    for subject in common_subjects:
        # Extract MMN data for the time window
        case1_data = case1_dict[subject].copy().crop(tmin=time_window[0]/1000, 
                                                     tmax=time_window[1]/1000)
        case2_data = case2_dict[subject].copy().crop(tmin=time_window[0]/1000, 
                                                     tmax=time_window[1]/1000)
        
        # Get mean amplitude for specific channels or all channels
        if channels:
            case1_amp = np.mean(case1_data.get_data(picks=channels), axis=1)
            case2_amp = np.mean(case2_data.get_data(picks=channels), axis=1)
        else:
            case1_amp = np.mean(case1_data.get_data(), axis=1)
            case2_amp = np.mean(case2_data.get_data(), axis=1)
        
        amplitudes_case1.append(case1_amp)
        amplitudes_case2.append(case2_amp)
    
    return np.array(amplitudes_case1), np.array(amplitudes_case2), common_subjects