# obtaining the receptive field of neurons using spike triggered averge
import os
import matplotlib.pyplot as plt
import numpy as np

from predusion.agent import Agent
import predusion.immaker as immaker

from kitti_settings import *

##### load the prednet
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

sub = Agent()
sub.read_from_json(json_file, weights_file)


###### generate images
n_image = 50 # number of images per trial
batch_size = 50 # number of trials
imshape = (128, 160)

color_w_im = immaker.Batch_gen().color_noise_full(imshape, n_image, batch_size)
grey_w_im = immaker.Batch_gen().grey_noise_full(imshape, n_image, batch_size)
color_w_im = grey_w_im

########### Prediction
#output_mode = 'prediction'
#out_im = sub.output(color_w_im, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)
#
##### plot out the prediction
#from predusion.ploter import Ploter
#
#fig, gs = Ploter().plot_seq_prediction(color_w_im[0], out_im[0])
#plt.show()

##### Record the neural activity
n_tao = min(10, n_image) # the number of of correlation time steps you want to see
n_eg_neuron = 5
crop_start = 2

output_mode = 'E0'
output = sub.output(color_w_im, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)

##### Processing images
color_w_im_proc = color_w_im[:, crop_start:] # cur the first few data
color_w_im_proc = color_w_im_proc.reshape(-1, *color_w_im_proc.shape[2:]) # combine batches. The final shape is [n_time, *imageshape, 3]

output_proc = output.reshape(output.shape[0], output.shape[1], -1) # flatten neural id
output_proc = output_proc[:, crop_start:] # crop the first few data
eg_neuron_id = np.linspace(0, output.shape[-1], n_eg_neuron, endpoint=False).astype(int) # uniformly sampled from all neurons
output_proc = output_proc[:, :, eg_neuron_id].reshape(-1, n_eg_neuron) # combine the batches. The final shape is [n_time, n_neurons]
output_proc = output_proc[n_tao:, :] # avoid end effect

output_n_time = output_proc.shape[0]
qij = [] # the correlation function
for tao in range(n_tao):
    rf_tao = np.tensordot(output_proc, color_w_im_proc[n_tao - tao:n_tao - tao + output_n_time], axes=([0], [0])) / output_n_time
    qij.append(rf_tao.copy())
qij = np.array(qij) # [number of tao time point, number of neurons, *image_shape, 3]

print(qij.shape)


plt.figure(1)
for i in range(1, n_tao + 1):
    plt.subplot(3, 4, i)
    plt.imshow(qij[i - 1, 0, :, :, 0])
plt.show()

#from predusion.ploter import Ploter
#Ploter.plot_seq(qij[:, 1, :, :, 0].astype(np.uint8))
#plt.show()
