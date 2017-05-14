'''
Train PredNet on KITTI sequences.
(Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)  # NOQA
from data_utils import SequenceGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Flatten
from keras.layers import TimeDistributed
from keras.models import Model
from prednet import PredNet
import kitti_settings


# Save the weights
save_model = True
# Weights location
weights_file = os.path.join(kitti_settings.WEIGHTS_DIR,
                            'prednet_kitti_weights.hdf5')
json_file = os.path.join(kitti_settings.WEIGHTS_DIR,
                         'prednet_kitti_model.json')

# Data files
train_file = os.path.join(kitti_settings.DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(kitti_settings.DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(kitti_settings.DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(kitti_settings.DATA_DIR, 'sources_val.hkl')

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
nt = 10
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_dim_ordering() == 'th' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1. / (nt - 1) * np.ones((nt, 1))
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
# errors will be (batch_size, nt, nb_layers)
errors = prednet(inputs)
# calculate weighted loss
errors_by_time = TimeDistributed(
    Dense(
        1, weights=[layer_loss_weights, np.zeros(1)], trainable=False),
    trainable=False)(errors)
# will be (batch_size, nt)
errors_by_time = Flatten()(errors_by_time)
final_errors = Dense(
    1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(
        errors_by_time)  # weight errors by time
model = Model(input=inputs, output=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(
    train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(
    val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)


def lr_schedule(epoch):
    if epoch < 75:
        return 0.001
    return 0.0001

callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(kitti_settings.WEIGHTS_DIR):
        os.mkdir(kitti_settings.WEIGHTS_DIR)
    callbacks.append(
        ModelCheckpoint(
            filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(
    train_generator,
    samples_per_epoch,
    nb_epoch,
    callbacks=callbacks,
    validation_data=val_generator,
    nb_val_samples=N_seq_val)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
