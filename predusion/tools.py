import numpy as np

def tensor_sta(fir_rate, stimuli, n_tao):
    '''
    spikes triggered average.
    input:
      fir_rate (array like, float, [n_time_step, ...])
      stimuli (array like, float, [n_time_step, ...]): should have the same n_time_step as fir_rate
      n_tao (int): how long the sta you want
    output:
      qij (numpy array, float, [n_tao, *fir_rate.shape[1:], *stimuli.shape[1:]]): the tensor dot of the fir_rate and stimuli
    '''
    fir_rate_proc = fir_rate[n_tao:] # avoid end effect
    fir_rate_proc = fir_rate_proc / np.sum(fir_rate_proc, axis=0)
    fir_rate_n_time = fir_rate_proc.shape[0]
    qij = [] # the correlation function
    for tao in range(n_tao):
        rf_tao = np.tensordot(fir_rate_proc, stimuli[n_tao - tao:n_tao - tao + fir_rate_n_time], axes=([0], [0]))
        qij.append(rf_tao.copy())
    qij = np.array(qij) # [number of tao time point, number of neurons, *image_shape, 3]
    return qij
