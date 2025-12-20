# create all the function for the ploting different kind of figure like 
#ERP plot   , topo plot ,
# graph plot
import matplotlib.pyplot as plt
import mne

def plot_erps(evoked_dict, picks, stylesdict, text_labels=None, title='', ylim=(-6.2, 6.2), time_range=(-0.1, 0.7), ylabel='Evoked Responses (µV)', dpi=100,ci=True,axs = None,color = None):
    """
    Plot personalized ERP comparisons using MNE's plot_compare_evokeds with consistent formatting.

    Parameters:
    - evoked_dict: dict, keys are condition names, values are lists of evoked objects
    - picks: list of channels (e.g., ['F3', 'Cz', 'F4']), only one pick will be used per plot
    - stylesdict: dict, keys must match evoked_dict keys; values are style dicts (linewidth, linestyle)
    - colors: list of colors for each condition (must match order of evoked_dict)
    - text_labels: list of (y_position, label_text), optional, for manual line legend
    - title: string, title of the plot
    - ylim: dict or tuple, y-axis limits
    - time_range: tuple, time cropping range (e.g., (-0.1, 0.7))
    - ylabel: string, y-axis label
    - dpi: int, resolution of figure

    Returns:
    - fig, axs: matplotlib Figure and Axes objects
    """

    # Crop each evoked list to specified time range
    evokeds_subset = {}
    for cond in evoked_dict:
        evoked_crop  = [evk.copy().crop(tmin=time_range[0], tmax=time_range[1]) for evk in evoked_dict[cond]]
        evokeds_subset[cond]  =evoked_crop 
    
    # Plot
    
    if axs is None:
        fig, axs = plt.subplots(figsize=(6, 5), dpi=100)
        return_fig = True
    else:
        return_fig = False
    mne.viz.plot_compare_evokeds(
        evokeds_subset,
        picks=picks,  # Usually 'Cz'
        colors= color,
        show_sensors=False,
        legend=False,
        styles=stylesdict,
        combine='mean',
        axes=axs,
        ylim=dict(eeg=ylim),
        time_unit='ms',
        truncate_xaxis=False,
        truncate_yaxis=False,
        show=False,
        title=title,ci=ci,
    )

    # Optional manual line labels
    if text_labels:
        for y, txt,col in text_labels:
            axs.text(0.75, y, txt, transform=axs.transAxes, fontsize=18,color=col)

    axs.set_ylabel(ylabel)
    #axs.set_xticks(np.arange(0, time_range[1]*1000 + 100, 200))  # Ticks every 0.5 units
    #axs.set_yticks(np.arange(-1, 1.1, 0.2)) # Ticks every 0.2 units
    # axs.grid(which='major', linestyle='-')

    # # Enable minor ticks and grid
    # axs.minorticks_on()
    # axs.grid(which='minor', linestyle=':')
    # Set consistent font sizes
    plt.rcParams.update({
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 18,
    })

    plt.tight_layout()
    
    if return_fig:
        return fig, axs
    
    else:
        return axs
