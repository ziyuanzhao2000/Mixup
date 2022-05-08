import numpy as np

window_length = 178
tr_in = np.load('full/train_input.npy')
te_in = np.load('full/test_input.npy')

tr_in = tr_in[:,0:1,:window_length]
te_in = te_in[:,0:1,:window_length]

np.save('train_input.npy', tr_in)
np.save('test_input.npy', te_in)
