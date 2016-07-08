'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from kers.engine.training import Model
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from process_kitti import SequenceGenerator

plot_save_dir = './kitti_plots/'
n_plot = 20

weights_file = 'prednet_weights.hdf5'
config_file = 'prednet_config.pkl'
test_file = './kitti_data/X_test.hkl'
test_sources = './kitti_data/sources_test.hkl'

batch_size = 5
nt = 10

# Load trained model
config = cPickle.load(open(config_file))
train_model = Model.from_config(config, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
inputs = Input(shape=train_model.layers[0].batch_input_shape[1:])
predictions = test_prednet(inputs)
test_model = Model(input=inputs, output=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique')
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)

# Compare MSE of PredNet predictions vs. using last frame.
mse_model = np.mean( (X[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X[:, :-1] - X[:, 1:])**2 )
print "Model MSE: %f" % mse_model
print "Previous Frame MSE: %f" % mse_prev

# Plot some predictions
aspect_ratio = float(X_hat.shape[3]) / X_hat.shape[4]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0.025, hspace=0.05)
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
for i in range(n_plot):
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(np.transpose(X_test[i,t], (1, 2, 0)), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual')

        plt.subplot(gs[t + nt])
        plt.imshow(np.transpose(X_hat[i,t], (1, 2, 0)), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted')

    plt.savefig(plot_save_dir + 'plot_' + str(i) + '.png')
    plt.clf()
