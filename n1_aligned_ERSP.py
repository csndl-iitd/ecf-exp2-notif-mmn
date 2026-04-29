import os
import glob
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from tqdm.auto import tqdm

# ==========================================
# 1. Directories & Configurations
# ==========================================
input_dir = "/Users/archinarang/Desktop/NeuroProject/gamma_preprocessed" 
output_dir = "/Users/archinarang/Desktop/NeuroProject/Broadband_ERSPs_N1_Aligned"
latency_dir = "/Users/archinarang/Desktop/NeuroProject/N1_latencies" 

os.makedirs(output_dir, exist_ok=True)

# Broadband Frequencies: 3 to 60 Hz
freqs = np.arange(3, 61, 1) 
freq_split_idx = np.where(freqs == 30)[0][0]

n_cycles = np.where(freqs < 30, freqs / 2.0, 7.0)

roi_configs = {
    'Temporal': ['T7', 'T8', 'CP5', 'CP6', 'C5', 'C6', 'FC5', 'FC6'],
    'Fronto_Central': ['Fz', 'Cz', 'FCz', 'F1', 'F2'],
    'Central_Parietal': ['Cz', 'CPz', 'Pz', 'CP1', 'CP2']
}

conds = {'15': 'Deviant Own', '25': 'Deviant Beep', 
         '14': 'Standard Own', '24': 'Standard Beep'}

# ======================
# 2. Load Latency Data 
# ======================
print("Loading and filtering N1 Latency Data...")
df_dev = pd.read_csv(os.path.join(latency_dir, "scores_n1_DEV_fronto_central_midline.csv"))
df_std = pd.read_csv(os.path.join(latency_dir, "scores_n1_STD_fronto_central_midline.csv"))

mean_lat_dev = df_dev.groupby(['subject', 'stim'])['PL'].mean().to_dict()
mean_lat_std = df_std.groupby(['subject', 'stim'])['PL'].mean().to_dict()

# Mean Inputation for outliers
LATENCY_CUTOFF = 0.400

# 1. Calculate the true group mean using ONLY healthy latencies
all_raw_latencies = list(mean_lat_dev.values()) + list(mean_lat_std.values())
healthy_latencies = [lat for lat in all_raw_latencies if lat <= LATENCY_CUTOFF]
imputed_mean = np.mean(healthy_latencies)

# 2. Replace the outliers with the imputed mean
outlier_count = 0
for key, lat in mean_lat_dev.items():
    if lat > LATENCY_CUTOFF:
        mean_lat_dev[key] = imputed_mean
        outlier_count += 1
        
for key, lat in mean_lat_std.items():
    if lat > LATENCY_CUTOFF:
        mean_lat_std[key] = imputed_mean
        outlier_count += 1

if outlier_count > 0:
    print(f"\nWARNING: Found {outlier_count} N1 latencies > {LATENCY_CUTOFF}s.")
    print(f"Action Taken: Replaced extreme outliers with the healthy group mean shift ({imputed_mean:.3f}s) to save subjects.\n")

# Calculate max shift for the safe tmax window using the corrected pool
all_corrected_latencies = list(mean_lat_dev.values()) + list(mean_lat_std.values())
max_latency = max(all_corrected_latencies)

# dynamic_tmax = Raw End (0.9s) - Max Shift - Edge Buffer (0.1s)
dynamic_tmax = 0.9 - max_latency - 0.1
dynamic_tmax = np.floor(dynamic_tmax * 1000) / 1000.0

print(f"Maximum Corrected N1 Latency: {max_latency:.3f}s")
print(f"Dynamically set Safe tmax for plotting: {dynamic_tmax:.3f}s")


known_subjects = list(set([k[0] for k in mean_lat_dev.keys()]))

# ==========================================
# 3. Main Loop
# ==========================================
for roi_name, chans in roi_configs.items():
    print(f"\n{'='*60}\nProcessing ROI: {roi_name}\n{'='*60}")
    
    file_list = glob.glob(os.path.join(input_dir, '*epo.fif'))
    group_data = {code: [] for code in conds.keys()}
    
    target_length = None 
    
    for file_path in tqdm(file_list, desc=f"Extracting Data ({roi_name})"):
        
        subj_id = next((s for s in known_subjects if s in os.path.basename(file_path)), None)
        if not subj_id:
            continue

        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        if not all(code in epochs.event_id for code in conds.keys()):
            continue
            
        epochs.pick(chans, verbose=False)
        
        for code in conds.keys():
            cond_name = conds[code]
            
            ###
            is_dev = 'Deviant' in cond_name
            stim_type = 'SN' if 'Own' in cond_name else 'BN'
            
            latency_dict = mean_lat_dev if is_dev else mean_lat_std
            shift_val = latency_dict.get((subj_id, stim_type))
            
            cond_epochs = epochs[code].copy()
            if shift_val is not None:
                
                sfreq = cond_epochs.info['sfreq']
                shift_val = np.round(shift_val * sfreq) / sfreq
                cond_epochs.shift_time(-shift_val, relative=True)

            # Transformation
            tfr = mne.time_frequency.tfr_morlet(
                cond_epochs, freqs=freqs, n_cycles=n_cycles, 
                return_itc=False, average=False, decim=3, n_jobs=-1, verbose=False
            )
            
            data = tfr.data.mean(axis=1) 
            times = tfr.times
            
            # Dynamic Baseline Tracking
            actual_shift = shift_val if shift_val is not None else 0.0
            base_start = -actual_shift - 0.2
            base_end = -actual_shift
            idx_base = np.where((times >= base_start) & (times <= base_end))[0]
            
            data_log = 10 * np.log10(data + 1e-10)
            base_log = data_log[:, :, idx_base].mean(axis=2, keepdims=True)
            
            trial_db = data_log - base_log
            
            
            
            start_idx = np.argmin(np.abs(times - (-0.2)))
            
            
            if target_length is None:
                end_idx = np.argmin(np.abs(times - dynamic_tmax))
                target_length = end_idx - start_idx + 1
                
           
            subj_avg_db = trial_db.mean(axis=0)[:, start_idx : start_idx + target_length]
            
            group_data[code].append(subj_avg_db)
            cropped_times = times[start_idx : start_idx + target_length] 

    # Convert to 3D Arrays
    for code in group_data.keys():
        group_data[code] = np.array(group_data[code])

    extent = [cropped_times[0], cropped_times[-1], freqs[0], freqs[-1]]

    # ==========================================
    # 4. PLOT 1 - PAIRED COMPARISONS
    # ==========================================
    print("Running Split-Mask Statistics & Generating Comparisons...")
    comparisons = {
        'Deviant Beep vs Standard Beep': group_data['25'] - group_data['24'],
        'Deviant Own vs Standard Own': group_data['15'] - group_data['14'],
        'Deviant Own vs Deviant Beep': group_data['15'] - group_data['25'],
        'Standard Own vs Standard Beep': group_data['14'] - group_data['24']
    }

    fig_comp, axes_comp = plt.subplots(2, 2, figsize=(16, 12))
    axes_comp = axes_comp.flatten()

    for i, (comp_title, diff_array) in enumerate(comparisons.items()):
        
        t_obs_low, cl_low, pv_low, _ = permutation_cluster_1samp_test(
            diff_array[:, :freq_split_idx, :], n_permutations=1000, out_type='mask', tail=0, n_jobs=-1, verbose=False
        )
        mask_low = np.zeros((freq_split_idx, diff_array.shape[2]), dtype=bool)
        if cl_low:
            for c_idx in np.where(pv_low < 0.05)[0]: mask_low = np.logical_or(mask_low, cl_low[c_idx])

        t_obs_high, cl_high, pv_high, _ = permutation_cluster_1samp_test(
            diff_array[:, freq_split_idx:, :], n_permutations=1000, out_type='mask', tail=0, n_jobs=-1, verbose=False
        )
        mask_high = np.zeros((diff_array.shape[1] - freq_split_idx, diff_array.shape[2]), dtype=bool)
        if cl_high:
            for c_idx in np.where(pv_high < 0.05)[0]: mask_high = np.logical_or(mask_high, cl_high[c_idx])

        full_mask = np.vstack((mask_low, mask_high))
        
        grand_avg_diff = diff_array.mean(axis=0)
        sig_diff_map = grand_avg_diff.copy()
        sig_diff_map[~full_mask] = np.nan  
        
        
        im = axes_comp[i].imshow(sig_diff_map, aspect='auto', origin='lower', 
                                 extent=extent, cmap='RdBu_r', 
                                 norm=TwoSlopeNorm(vmin=-0.5, vcenter=0., vmax=1.5))
        
        
        axes_comp[i].set_xlim(-0.2, dynamic_tmax) 
        axes_comp[i].set_facecolor('white')
        axes_comp[i].set_title(comp_title, fontsize=16, fontweight='bold')
        axes_comp[i].set_ylabel("Freq (Hz)", fontsize=12)
        axes_comp[i].set_xlabel("Time (s)", fontsize=12)
        
        divider = make_axes_locatable(axes_comp[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Group Avg dB Diff')

    plt.suptitle(f"{roi_name.replace('_', ' ')}: PAIRED DIFFERENCES N1-ALIGNED (3-60 Hz)\n(p < 0.05)", fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    save_path_comp = os.path.join(output_dir, f"{roi_name}_Comparisons.png")
    plt.savefig(save_path_comp, dpi=300)
    plt.close(fig_comp)

    # ==========================================
    # 5. PLOT 2 - BASELINE ACTIVATIONS
    # ==========================================
    print("Running Split-Mask Statistics & Generating Baselines...")
    baseline_conds = {
        'Deviant Own': group_data['15'],
        'Deviant Beep': group_data['25'],
        'Standard Own': group_data['14'],
        'Standard Beep': group_data['24']
    }

    fig_base, axes_base = plt.subplots(2, 2, figsize=(16, 12))
    axes_base = axes_base.flatten()

    for i, (cond_title, data_array) in enumerate(baseline_conds.items()):
        
        t_obs_low, cl_low, pv_low, _ = permutation_cluster_1samp_test(
            data_array[:, :freq_split_idx, :], n_permutations=1000, out_type='mask', tail=0, n_jobs=-1, verbose=False
        )
        mask_low = np.zeros((freq_split_idx, data_array.shape[2]), dtype=bool)
        if cl_low:
            for c_idx in np.where(pv_low < 0.05)[0]: mask_low = np.logical_or(mask_low, cl_low[c_idx])

        t_obs_high, cl_high, pv_high, _ = permutation_cluster_1samp_test(
            data_array[:, freq_split_idx:, :], n_permutations=1000, out_type='mask', tail=0, n_jobs=-1, verbose=False
        )
        mask_high = np.zeros((data_array.shape[1] - freq_split_idx, data_array.shape[2]), dtype=bool)
        if cl_high:
            for c_idx in np.where(pv_high < 0.05)[0]: mask_high = np.logical_or(mask_high, cl_high[c_idx])

        full_mask = np.vstack((mask_low, mask_high))
        
        grand_avg_db = data_array.mean(axis=0)
        sig_map = grand_avg_db.copy()
        sig_map[~full_mask] = np.nan  
        
        
        im = axes_base[i].imshow(sig_map, aspect='auto', origin='lower', 
                                 extent=extent, cmap='RdBu_r', 
                                 norm=TwoSlopeNorm(vmin=-0.5, vcenter=0., vmax=1.5))
        
        
        axes_base[i].set_xlim(-0.2, dynamic_tmax)
        axes_base[i].set_facecolor('white')
        axes_base[i].set_title(cond_title, fontsize=16, fontweight='bold')
        axes_base[i].set_ylabel("Freq (Hz)", fontsize=12)
        axes_base[i].set_xlabel("Time (s)", fontsize=12)
        
        divider = make_axes_locatable(axes_base[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='dB vs Baseline')

    plt.suptitle(f"{roi_name.replace('_', ' ')}: VS BASELINE N1-ALIGNED (3-60 Hz)\n(p < 0.05)", fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    save_path_base = os.path.join(output_dir, f"{roi_name}_Baseline.png")
    plt.savefig(save_path_base, dpi=300)
    plt.close(fig_base)

print("\nBroadband analysis complete! All final N1-aligned maps generated.")