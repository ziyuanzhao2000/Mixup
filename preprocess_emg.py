import numpy as np
import os
import pickle
import wfdb
import csv
from tqdm import tqdm

alias = 'emg'
data_folder = 'data'
basepath = os.path.join(os.getcwd(), data_folder, alias)
data_file_names = ['emg_healthy.txt', 'emg_myopathy.txt', 'emg_neuropathy.txt']
labels = [0, 1, 2]

sample_fs = 4000 # Hz
target_fs = 4000 # Hz
downsample_stride = sample_fs // target_fs
target_fs = sample_fs // downsample_stride
window_len = 1500 # samples

# first stage, read in datasets and make the dictionaries for X (id -> time series) and y (id -> patient no.)
input_train = {}
output_train = {}
input_test = {}
output_test = {}
pid = 0
n_samples = 204
train_proportion = 0.8
train_set = np.random.choice(np.arange(n_samples), int(train_proportion * n_samples), replace=False)


for data_file_name, label in zip(data_file_names, labels):
    signal = np.loadtxt(os.path.join(basepath, data_file_name))[:,1:2] # first column is timestamp, not needed
    signal = signal[::downsample_stride,:] # downsample
    signal_length = signal.shape[0]
    signal = signal[:signal_length // window_len * window_len,:]
    signal_length = signal.shape[0]
    signals = signal.reshape((signal_length // window_len, window_len))
    for i in range(signals.shape[0]):
        if pid in train_set:
            input_train[pid] = signals[i:i+1, :]
            output_train[pid] = label
        else:
            input_test[pid] = signals[i:i+1, :]
            output_test[pid] = 2 #np.random.randint(0, 3)
        pid += 1

for input, output, partition in zip([input_train, input_test], [output_train, output_test], ['train', 'test']):
    input_arr = np.vstack(tuple(input.values()))
    input_arr = np.expand_dims(input_arr, axis=1)
    input_nrows = [arr.shape[0] for arr in input.values()]
    output_replicated = [[label] * n for (n, label) in zip(input_nrows, output.values())]
    output_arr = [el for sublst in output_replicated for el in sublst]
    output_arr = np.expand_dims(output_arr, axis=1)

    np.save(os.path.join(basepath, f"{partition}_input_{window_len}"), input_arr)
    np.save(os.path.join(basepath, f"{partition}_output_{window_len}"), output_arr)
