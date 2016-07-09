'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
from six.moves import cPickle

from keras import backend as K
from keras.engine.training import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from process_kitti import SequenceGenerator
import kitti_settings

save_model = True  # if weights will be saved
weights_file = os.path.join(weights_dir, 'prednet_kitti_weights.hdf5')  # where weights will be saved
config_file = os.path.join(weights_dir, 'prednet_kitti_config.pkl')

# Data files
train_file = os.path.join(data_dir, 'X_train.hkl')
train_sources = os.path.join(data_dir, 'sources_train.hkl')
val_file = os.path.join(data_dir, 'X_val.hkl')
val_sources = os.path.join(data_dir, 'sources_val.hkl')

# Training parameters
nb_epoch = 100
batch_size = 5
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

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
optimizer = Adam(lr=0.0005)
model.compile(loss='mean_absolute_error', optimizer=optimizer)

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)
callbacks = []
if save_model:
    if not os.path.exists(weights_dir): os.mkdir(weights_dir)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks,
                    validation_data=val_generator, nb_val_samples=N_seq_val)

if save_model:
    config = model.get_config()
    cPickle.dump(config, open(config_file, 'w'))
