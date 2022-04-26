import torch
import numpy as np
train_dict = torch.load('train.pt')
val_dict = torch.load('val.pt')
test_dict = torch.load('test.pt')

train_input = np.vstack([train_dict['samples'], val_dict['samples']])
test_input = test_dict['samples']
train_output = np.expand_dims(np.concatenate([train_dict['labels'], val_dict['labels']]), axis=1)
test_output = np.expand_dims(test_dict['labels'], axis=1)

np.save('train_input.npy', train_input)
np.save('test_input.npy', test_input)
np.save('train_output.npy', train_output)
np.save('test_output.npy', test_output)