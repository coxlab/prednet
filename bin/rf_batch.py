# obtaining the receptive field of neurons using spike triggered averge
import os
import matplotlib.pyplot as plt
import numpy as np

from predusion.tools import tensor_sta
from predusion.agent import Agent
import predusion.immaker as immaker

from kitti_settings import *

##### load the prednet
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

sub = Agent()
sub.read_from_json(json_file, weights_file)
print(sub.get_config())

###### generate images
n_image = 2 # number of images per trial
batch_size = 2 # number of trials
imshape = (128, 160)
n_tao = min(3, n_image) # the number of of correlation time steps you want to see
crop_start = 3
epoch = 1
output_mode = 'E1'

eg_neuron_id = [200] # the example neuron id, set it as None so that eg_neurons id would be linspcae(all neurons, n_eg_neuron)
n_eg_neuron = 1
show_neuron_id = 0 # show the ith example neuron in the eg_neuron_id
sig = (1, 1, 1)

for ep in range(epoch):
    color_w_im = immaker.Batch_gen().color_noise_full(imshape, n_image, batch_size, sig=sig)
    #grey_w_im = immaker.Batch_gen().grey_noise_full(imshape, n_image, batch_size)
    #color_w_im = grey_w_im

    ##### Record the neural activity

    output = sub.output(color_w_im, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (batch_size, number of images in a seq, a 3d tensor represent neural activation)

    ##### Processing images
    color_w_im_proc = color_w_im[:, crop_start:] # cur the first few data
    color_w_im_proc = color_w_im_proc.reshape(-1, *color_w_im_proc.shape[2:]) # combine batches. The final shape is [n_time, *imageshape, 3]

    output_proc = output.reshape(output.shape[0], output.shape[1], -1) # flatten neural id
    output_proc = output_proc[:, crop_start:] # crop the first few data
    if eg_neuron_id is None:
        eg_neuron_id = np.linspace(0, output.shape[-1], n_eg_neuron, endpoint=False).astype(int) # uniformly sampled from all neurons
    else:
        n_eg_neuron = len(eg_neuron_id)

    #output_proc_neuron = output_proc.reshape(-1, output_proc.shape[-1]) # (firing rate of neuron, neuron id)
    #non_trivial_neuron_id = np.linalg.norm(output_proc_neuron, axis=0) > 1e-10 # many neurons have zero firing rate, we only pick the ones has non-zero firing rate.
    #output_proc = output_proc[:, :, non_trivial_neuron_id]

    output_proc = output_proc[:, :, eg_neuron_id].reshape(-1, n_eg_neuron) # combine the batches. The final shape is [n_time, n_neurons]

    color_w_im_proc = color_w_im_proc / 255 # rescale the image
    qij = tensor_sta(output_proc, color_w_im_proc, n_tao)

    if ep == 0:
        qtot = qij
    else:
        qtot = qij + qtot

    print('epoch = {}'.format(ep))

qtot = qtot / epoch

img_range = [128, 160]
fig = plt.figure(1)
qij_flat = qij[1-1:n_tao, show_neuron_id, :img_range[0], :img_range[1], :].flatten()
vmin, vmax = np.min(qij_flat), np.max(qij_flat)
for i in range(1, n_tao + 1):
    plt.subplot(3, 3, i)
    plt.imshow(qij[i - 1, show_neuron_id, :img_range[0], :img_range[1], 0], vmin=vmin, vmax=vmax, cmap='viridis') # R channel
    plt.subplot(3, 3, i + 3)
    plt.imshow(qij[i - 1, show_neuron_id, :img_range[0], :img_range[1], 1], vmin=vmin, vmax=vmax, cmap='viridis') # G channel
    plt.subplot(3, 3, i + 6)
    plt.imshow(qij[i - 1, show_neuron_id, :img_range[0], :img_range[1], 2], vmin=vmin, vmax=vmax, cmap='viridis') # B channel
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig('./figs/rf.pdf', format='pdf', dpi=1000)
plt.show()

#from predusion.ploter import Ploter
#Ploter.plot_seq(qij[:, 1].astype(np.uint8))
#plt.show()
