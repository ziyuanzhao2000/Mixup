import numpy as np
import os
import pickle
import wfdb
import csv
from tqdm import tqdm

alias = 'training2017'
basepath = f'{os.getcwd()}/data/{alias}'
sampling_fs = 300 # Hz
window_len = 1500 # samples

# First get list of file names from the data folder
file_names = [file_name.split('.hea')[0] for file_name in os.listdir(basepath) if '.hea' in file_name]
diagnoses = []
with open(os.path.join(basepath, 'REFERENCE.csv'), 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        diagnoses.append(line[1])

# Process each file by dividing up time series into contiguous windows
input = {}
output = {}
for file_name, diagnosis in tqdm(zip(file_names, diagnoses)):
    signal, labels = wfdb.rdsamp(os.path.join(basepath, file_name))

    # clip the tail that is shorter than a minimal window, then reshape
    signal_len = signal.shape[0] // window_len * window_len
    signal = signal[:signal_len].reshape((signal_len // window_len, window_len))
    input[file_name] = signal
    assert(signal.shape[1] == window_len and signal.shape[0] > 0 and signal.ndim == 2)

    if diagnosis == "N":
        output[file_name] = 0
    elif diagnosis == "A":
        output[file_name] = 1
    elif diagnosis == "O":
        output[file_name] = 2
    elif diagnosis == "~":
        output[file_name] = 3
    else:
        output[file_name] = -1


input_arr = np.vstack(tuple(input.values()))
input_arr = np.expand_dims(input_arr, axis=1)
input_nrows = [arr.shape[0] for arr in input.values()]
output_replicated = [[label] * n for (n, label) in zip(input_nrows, output.values())]
output_arr = [el for sublst in output_replicated for el in sublst]
output_arr = np.expand_dims(output_arr, axis=1)

np.save(os.path.join(basepath, f"input_{window_len}"), input_arr)
np.save(os.path.join(basepath, f"output_{window_len}"), output_arr)

