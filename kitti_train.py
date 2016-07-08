'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import numpy as np
from six.moves import cPickle

from keras import backend as K
from keras.engine.training import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint

from prednet import PredNet
from process_kitti import SequenceGenerator

# Training parameters
nb_epoch = 2
batch_size = 5
samples_per_epoch = 100 #500
N_seq_val = 100  # number of sequences to use for validation
use_early_stopping = True
patience = 5
save_model = True
save_name = 'prednet_'

# Data files
train_file = './kitti_data/X_train.hkl'
train_sources = './kitti_data/sources_train.hkl'
val_file = './kitti_data/X_val.hkl'
val_sources = './kitti_data/sources_val.hkl'

# Model parameters
nt = 10
input_shape = (3, 128, 160)
stack_sizes = (input_shape[0], 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(input=inputs, output=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)
callbacks = []
if use_early_stopping:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=patience))
if save_model:
    callbacks.append(ModelCheckpoint(filepath=save_name + 'weights.hdf5', monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks,
                              validation_data=val_generator, nb_val_samples=val_generator.N_sequences/batch_size)

if save_model:
    config = model.get_config()
    cPickle.dump(config, open(save_name + '_config.pkl', 'w'))

#TODO:  remove
from helpers import plot_training_error
plot_training_error(train_err = history.history['loss'], val_err = history.history['val_loss'], run_name = 'kitti', out_file = 'error_plot.jpg')
