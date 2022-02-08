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
n_deg = 20
n_repeat = 5
center = 10
width = int(imshape[0] / 2)
bg = 150 # backgound color for the square
neuron_t = 3 # obtain neural activity at the 4th time step
eg_neuron_id = 10 # example neuron id
eg_color_id = 10

deg_color = Degree_color(center_l=l, center_a=a, center_b=b, radius=r)


degree = np.linspace(0, 360, n_deg)
color_list = deg_color.out_color(degree, fmat='RGB', is_upscaled=True)
square = immaker.Square(imshape, background=bg)
seq_gen = immaker.Seq_gen()

#### Start testing
eg_tuning = [] # collecting response of one example neuron
eg_color_i = -1

for color in color_list:
    eg_color_i += 1

    r, g, b = color
    im = square.set_full_square(color=color)
    seq_repeat = seq_gen.repeat(im, n_repeat)

    #seq_pred = sub.output(seq_repeat)

    ##### plot out the prediction
    #f, ax = plt.subplots(2, n_repeat, sharey=True, sharex=True)
    #for i, sq_p, sq_r in zip(range(n_repeat), seq_pred, seq_repeat):
    #    ax[0][i].imshow(sq_r.astype(int))
    #    ax[1][i].imshow(sq_p.astype(int))

    #plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    #plt.show()


    ##### collecting tuning data and example dynamics
    r1 = sub.output(seq_repeat, output_mode='R1') # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)
    r1 = r1.reshape(r1.shape[0], -1)

    # grap one example neuron at time = 5
    eg_r1 = r1[neuron_t, eg_neuron_id]
    eg_tuning.append(eg_r1)

    ##### collecting example neural dynamics to one color

plt.figure()
plt.plot(degree, eg_tuning)
plt.show()
