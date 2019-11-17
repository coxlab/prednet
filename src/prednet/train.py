
import os.path
import numpy as np
import prednet.prednet
import prednet.data_utils
import keras.layers
import keras.callbacks
import keras.models
import keras.backend


def train_on_hickles(DATA_DIR,
                     number_of_epochs=150, steps_per_epoch=125,
                     path_to_save_model_json='prednet_model.json', path_to_save_weights_hdf5='prednet_weights.hdf5',
                     ):
  # Data files
  train_file = os.path.join(DATA_DIR, 'X_train.hkl')
  train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
  val_file = os.path.join(DATA_DIR, 'X_validate.hkl')
  val_sources = os.path.join(DATA_DIR, 'sources_validate.hkl')
  return train_on_arrays_and_sources(train_file, train_sources, val_file, val_sources,
                                     path_to_save_model_json, path_to_save_weights_hdf5,
                                     number_of_epochs, steps_per_epoch)


def train_on_arrays_and_sources(train_file, train_sources, val_file, val_sources,
                                path_to_save_model_json='prednet_model.json', path_to_save_weights_hdf5='prednet_weights.hdf5',
                                number_of_epochs=150, steps_per_epoch=125,
                                ):

  save_model = True  # if weights will be saved

  # Training parameters
  batch_size = 4
  samples_per_epoch = steps_per_epoch * batch_size
  N_seq_val = 100  # number of sequences to use for validation

  nt = 8  # number of timesteps used for sequences in training

  train_generator = prednet.data_utils.SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
  val_generator = prednet.data_utils.SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)
  assert train_generator.im_shape == val_generator.im_shape

  # Model parameters
  n_channels = 3
  # input_shape = (n_channels, im_height, im_width) if keras.backend.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
  # assert input_shape == train_generator.im_shape
  stack_sizes = (n_channels, 48, 96, 192)
  R_stack_sizes = stack_sizes
  A_filt_sizes = (3, 3, 3)
  Ahat_filt_sizes = (3, 3, 3, 3)
  R_filt_sizes = (3, 3, 3, 3)
  layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
  layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
  time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
  time_loss_weights[0] = 0

  predictor = prednet.prednet.PredNet(stack_sizes, R_stack_sizes,
                    A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                    output_mode='error', return_sequences=True)
  
  inputs = keras.layers.Input(shape=(nt,) + train_generator.im_shape)
  errors = predictor(inputs)  # errors will be (batch_size, nt, nb_layers)
  errors_by_time = keras.layers.TimeDistributed(keras.layers.Dense(1, trainable=False),
                                                weights=[layer_loss_weights, np.zeros(1)],
                                                trainable=False,
                                                )(errors)  # calculate weighted error by layer
  errors_by_time = keras.layers.Flatten()(errors_by_time)  # will be (batch_size, nt)
  final_errors = keras.layers.Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
  model = keras.models.Model(inputs=inputs, outputs=final_errors)
  model.compile(loss='mean_absolute_error', optimizer='adam')

  lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
  callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule)]
  if save_model:
      if not os.path.exists(os.path.dirname(path_to_save_weights_hdf5)):
        os.mkdirs(os.path.dirname(path_to_save_weights_hdf5))
      print('Setting keras.callbacks.ModelCheckpoint for', path_to_save_weights_hdf5)
      callbacks.append(keras.callbacks.ModelCheckpoint(filepath=path_to_save_weights_hdf5,
                                                       monitor='val_loss', save_best_only=True))
  else:
      raise NotImplementedError("It appears that evaluation requires the HDF5 file.")
  
  history = model.fit_generator(train_generator, steps_per_epoch, number_of_epochs,
                                callbacks=callbacks,
                  validation_data=val_generator, validation_steps=N_seq_val / batch_size)
  
  if save_model:
      if not os.path.exists(os.path.dirname(path_to_save_model_json)):
        os.mkdirs(os.path.dirname(path_to_save_model_json))
      json_string = model.to_json()
      with open(path_to_save_model_json, "w") as f:
          f.write(json_string)
      model.save_weights(path_to_save_weights_hdf5)
      assert os.path.exists(path_to_save_weights_hdf5)
