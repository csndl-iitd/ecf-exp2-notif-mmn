rawdir = '../data/raw/'
resultdir = '../data/results/'
tmpdir = '../data/tmp/'

stims = ['BN', 'SN', 'CN']
stim_types = ['STD', 'DEV']
triggers = {
    'BN STD': 'S14',
    'SN STD': 'S24',
    'CN STD': 'S34', # this is actually the BN tone
    'BN DEV': 'S25',
    'SN DEV': 'S15',
    'CN DEV': 'S35',
}
triggers_r = {v: k for k, v in triggers.items()}
locations = {
    'inverse_mastoid': ['T7', 'CP5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'CP6', 'T8'], # https://www.sciencedirect.com/science/article/pii/S0301051121000156
    'right_frontal': ['F4', 'F6', 'AF4', 'AF8', 'FC4', 'FC6'],
    'supratemporal': ['TP9', 'TP10', 'T7', 'T8', 'TP7', 'TP8'],
    'fronto_central_midline': ['Cz', 'Fz', 'FCz', 'FC1', 'FC2'],
}
sfreq = 512