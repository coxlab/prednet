import os
import matplotlib.pyplot as plt
import numpy as np

from predusion.agent import Agent
import predusion.immaker as immaker
from predusion.color_deg import Degree_color

from kitti_settings import *

##### load the prednet
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

sub = Agent()
sub.read_from_json(json_file, weights_file)

###### generate images
n_image, n_image_w = 4, 4

imshape = (128, 160)

l, a, b, r = 80, 22, 14, 52
n_deg = 20
n_repeat = 20
n_eg_neuron = 10 # number of example neurons
eg_color_id = 10

deg_color = Degree_color(center_l=l, center_a=a, center_b=b, radius=r)

degree = np.linspace(0, 360, n_deg)
color_list = deg_color.out_color(degree, fmat='RGB', is_upscaled=True)
square = immaker.Square(imshape)
seq_gen = immaker.Seq_gen()

##### Collecting one example dynamics
im = square.set_full_square(color=color_list[eg_color_id])
seq_repeat = seq_gen.repeat(im, n_repeat)

fea_map = sub.output(seq_repeat, output_mode='E0') # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)
fea_map = fea_map.reshape(fea_map.shape[0], -1)

eg_neuron_id = np.linspace(0, fea_map.shape[-1], n_eg_neuron, endpoint=False).astype(int) # uniformly sampled from all neurons

print(eg_neuron_id)

plt.figure()

for idx in eg_neuron_id:
    plt.plot(np.arange(n_repeat), fea_map[:, idx])

plt.xlabel('Time step')
plt.ylabel('Example neural activity')
plt.show()

#### plot out the prediction
import matplotlib.gridspec as gridspec

plt.figure(figsize = (n_repeat, 2))
gs = gridspec.GridSpec(2, n_repeat)
gs.update(wspace=0., hspace=0.)

seq_pred = sub.output(seq_repeat)

for t, sq_p, sq_r in zip(range(n_repeat), seq_pred, seq_repeat):
    plt.subplot(gs[t])
    plt.imshow(sq_r.astype(int))
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.subplot(gs[t + n_repeat])
    plt.imshow(sq_p.astype(int))
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

plt.show()
