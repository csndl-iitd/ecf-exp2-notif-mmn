# the excute the pipeline for all tha preprocessing 
import os
import sys
sys.path.append(os.path.abspath('..'))
from utils import remove_epochs_with_n_bad_channels, config,available_event_
from preprocessing import EEG_preprocessing
from erp import event_ERP


def run(config):
    for sub in os.listdir(config['data_folder']):
        path = os.path.join(config['data_folder'],sub,config['data_type'])
        cl = EEG_preprocessing(path,config) 
        event_ERP(sub,cl , config )
    return config