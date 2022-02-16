from matplotlib import pyplot as plt
import numpy as np

########## question 5 ##########
from scipy.io import loadmat
from scipy.linalg import toeplitz

m = loadmat('./bin/ganglion.mat')
stim = m['stim'][:, :, :400]
counts = m['counts'].flatten()[:400]
nsta = 12

nullArr = np.zeros(nsta)
nullArr[0] = counts[-1]
ttSpikes = np.sum(counts)
counts = counts / ttSpikes
countsMat = toeplitz(counts[::-1], nullArr)[::-1]
stavg = np.tensordot(stim, countsMat, axes = ([2], [0]))
plt.figure(0)
for i in range(1, nsta + 1):
    plt.subplot(3, 4, i)
    plt.imshow(stavg[:, :, i - 1])
plt.show()

#print(counts.shape)
#print(stim.shape)
n_tao = 12

from predusion.tools import tensor_sta

qij = tensor_sta(counts, stim.transpose([2, 0, 1]), n_tao)

#counts = counts[n_tao:]
#n_time = counts.shape[0]
#qij = []
#for tao in range(n_tao):
#    rf_tao = np.tensordot(counts, stim[:, :, n_tao - tao : n_tao - tao + n_time], axes=([0], [2])) / n_time
#    qij.append(rf_tao.copy())
#qij = np.array(qij) # [number of tao time point, number of neurons, *image_shape, 3]
#print(qij.shape)
#
plt.figure(1)
for i in range(1, nsta + 1):
    plt.subplot(3, 4, i)
    plt.imshow(qij[i-1, :, :])
plt.show()

