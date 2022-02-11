# obtain the tuning curve of neurons in the prednet, in terms of different stimuli color
# 1. generate input sequence
# 2. feed the input sequence to the PredNet
# 3. Obtain the feature map of the PredNet
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

l, a, b, r = 80, 22, 14, 70
n_deg = 50
n_repeat = 10
center = 10
width = int(imshape[0] / 2)
bg = 150 # backgound color for the square
neuron_t = 7 # obtain neural activity at the 4th time step
n_eg_neuron = 50 # example neuron id
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'tuning/')
batch_size = 20

if not os.path.exists(plot_save_dir): os.makedirs(plot_save_dir)

deg_color = Degree_color(center_l=l, center_a=a, center_b=b, radius=r)

degree = np.linspace(0, 360, n_deg)
color_list = deg_color.out_color(degree, fmat='RGB', is_upscaled=True)
square = immaker.Square(imshape, background=bg)
seq_gen = immaker.Seq_gen()

def get_runing_module(output_mode='E0'):
    '''
    get example tuning in a specific module
    input:
      name (str): module name
    output:
      eg_tuning
    '''
    seq_list = []
    for color in color_list:

        r, g, b = color
        im = square.set_full_square(color=color)
        seq_repeat = seq_gen.repeat(im, n_repeat)
        seq_list.append(seq_repeat)

    seq_list = np.array(seq_list)
    ##### collecting tuning data and example dynamics
    r1 = sub.output(seq_list, output_mode=output_mode, batch_size=batch_size) # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)
    r1 = r1.reshape(r1.shape[0], r1.shape[1], -1)

    eg_neuron_id = np.linspace(0, r1.shape[-1], n_eg_neuron, endpoint=False).astype(int) # uniformly sampled from all neurons
    eg_tuning = r1[:, neuron_t, eg_neuron_id]
    eg_tuning = np.array(eg_tuning).T

    all_tuning = r1[:, neuron_t, :]
    all_tuning = np.array(all_tuning).T

    return eg_tuning, all_tuning

n_layer = 4
output_mode_list = [key + str(l) for key in ['E', 'R', 'A', 'Ahat'] for l in range(n_layer)]

data = {}

for output_mode in output_mode_list:
    data[output_mode + '_eg'], data[output_mode + '_all'] = get_runing_module(output_mode=output_mode)

    plt.figure()
    for neuron in data[output_mode + '_eg']:
        plt.plot(degree, neuron)
    plt.title(output_mode)
    plt.savefig(plot_save_dir + output_mode  + '.png')
    print('{0} tuning saved'.format(output_mode))

print(sub.get_config()) # print out the config of the prednet
