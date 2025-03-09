import datetime
import os
import logging
from joblib import delayed
import pandas as pd
import numpy as np
import locale

from .raw_signal import RawSignal
from .raw_signals import RawSignals
from ..tools.progressparallel import ProgressParallel

date_format_string = "%Y-%m-%d %H:%M:%S"

def read_signals_from_dirs(input_dir, sample_rate=1000):
    """
     Reads raw signals from the directory structure.
     Return tuple of accepted and rejected signals
    """
    accepted  = _read_signals_from_dirs_internal(input_dir, sample_rate)

    rejected_measurements_path = os.path.join(input_dir,"rejected")
    if os.path.exists(rejected_measurements_path):
        rejected = _read_signals_from_dirs_internal(  rejected_measurements_path, sample_rate )
    else:
        rejected = None

    return {"accepted": accepted, "rejected": rejected}

def _read_class_dir(class_dir):
    """
    Read objects from class-specific directory
    Arguments:
     class_dir -- class specific directories. It contains csv and dat files
    """
    csv_files_list = [file for file in sorted(os.listdir(class_dir)) if file.endswith(".csv")]
    class_name = os.path.basename(class_dir)

    signal_objects = RawSignals()

    for file in csv_files_list:
        file_basename = os.path.splitext(file)[0]
        csv_path = os.path.join(class_dir,"{}.csv".format(file_basename))
        dat_path = os.path.join(class_dir,"{}.dat".format(file_basename))

        try:
            data = np.asfortranarray (pd.read_csv(csv_path, delimiter=';', decimal=',',header=None).to_numpy())
        except Exception as exc:
            logging.debug("Failed to load {}. Exception: {}. Skipping".format(csv_path,exc))
            continue

        object_timestamp = 0
        try:
            dat_handler = open(dat_path,"r")
            data_text = dat_handler.read().strip()
            element = datetime.datetime.strptime(data_text,date_format_string)
            object_timestamp = datetime.datetime.timestamp(element)
        except Exception as exc:
            logging.debug("Failed to determine timestamp for {}. Exception {}".format(csv_path, exc))

        signal_objects.append(RawSignal(data,class_name,timestamp=object_timestamp))
    
    return signal_objects

        

def _read_signals_from_dirs_internal(input_dir, sample_rate=1000):
    """
    Read the raw dataset from the directory structure.
    """
    sorted_class_dirs = sorted( [ d for d in os.listdir(os.path.normpath(input_dir)) 
            if  os.path.isdir( os.path.join(input_dir,d) ) and d != 'rejected'  ] )


    data_objects = RawSignals( sample_rate=sample_rate)

    if len(sorted_class_dirs) == 0:
            return data_objects
    
    class_data_objects = ProgressParallel(n_jobs=-1,use_tqdm=True,total=len(sorted_class_dirs),desc="Class directories")(delayed(_read_class_dir)(os.path.join(input_dir, directory)) 
                                        for directory in sorted_class_dirs )
    for class_data_obj in class_data_objects:
        data_objects+= class_data_obj

    return data_objects


def save_signals_to_dirs(raw_signals:RawSignals, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    signal_labels = raw_signals.get_labels()
    unique_labels = set(signal_labels)

    for label in unique_labels:
        label_dir_path = os.path.join(output_directory, "{}".format(label))
        os.makedirs(label_dir_path, exist_ok=True)

        label_indices = [ i for i in range(len(raw_signals)) if raw_signals[i].object_class == label ]
        signal_label_subset = raw_signals[label_indices]
        
        subset_signal_indices_string = sorted([ "{}".format(i) for i in range(1,len(signal_label_subset)+1)])

        cnt = 0
        for istr in subset_signal_indices_string:
            
            data_file_path = os.path.join(label_dir_path,"{}.csv".format(istr))
            
            signal_np = signal_label_subset[cnt].signal
            signal_df = pd.DataFrame(signal_np)
            signal_df.to_csv(data_file_path, sep=";",header=False,index=False, decimal=',')

            date_file_path = os.path.join(label_dir_path,"{}.dat".format(istr))
            date_object = datetime.datetime.fromtimestamp(signal_label_subset[cnt].timestamp)
            date_string = date_object.strftime(date_format_string)

            with open(date_file_path,"w") as file:
                print(date_string, file=file)
            cnt += 1

 