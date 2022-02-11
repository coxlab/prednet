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
n_image = 4 # number of images per trial
batch_size = 1 # number of trials
imshape = (128, 160)

color_w_im = immaker.Batch_gen().color_noise_full(imshape, n_image, batch_size)
grey_w_im = immaker.Batch_gen().grey_noise_full(imshape, n_image, batch_size)

########## Prediction
output_mode = 'prediction'
out_im = sub.output(color_w_im, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)

#### plot out the prediction
from predusion.ploter import Ploter

fig, gs = Ploter().plot_seq_prediction(color_w_im[0], out_im[0])
plt.show()
