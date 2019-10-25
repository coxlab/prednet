
import os.path
import numpy as np
import prednet.prednet
import prednet.data_utils
import keras.layers
import keras.callbacks
import keras.models
import keras.backend

def train_on_hickles(DATA_DIR, WEIGHTS_DIR, im_height, im_width, number_of_epochs=150):
  save_model = True  # if weights will be saved
  weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
  json_file = os.path.join(WEIGHTS_DIR, 'prednet_model.json')
  
  # Data files
  train_file = os.path.join(DATA_DIR, 'X_train.hkl')
  train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
  val_file = os.path.join(DATA_DIR, 'X_validate.hkl')
  val_sources = os.path.join(DATA_DIR, 'sources_validate.hkl')
  
  # Training parameters
  batch_size = 4
  samples_per_epoch = 500
  N_seq_val = 100  # number of sequences to use for validation
  
  # Model parameters
  n_channels = 3
  input_shape = (n_channels, im_height, im_width) if keras.backend.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
  stack_sizes = (n_channels, 48, 96, 192)
  R_stack_sizes = stack_sizes
  A_filt_sizes = (3, 3, 3)
  Ahat_filt_sizes = (3, 3, 3, 3)
  R_filt_sizes = (3, 3, 3, 3)
  layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
  layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
  nt = 10  # number of timesteps used for sequences in training
  time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
  time_loss_weights[0] = 0
  
  
  predictor = prednet.prednet.PredNet(stack_sizes, R_stack_sizes,
                    A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                    output_mode='error', return_sequences=True)
  
  inputs = keras.layers.Input(shape=(nt,) + input_shape)
  errors = predictor(inputs)  # errors will be (batch_size, nt, nb_layers)
  errors_by_time = keras.layers.TimeDistributed(keras.layers.Dense(1, trainable=False),
                                                weights=[layer_loss_weights, np.zeros(1)],
                                                trainable=False,
                                                )(errors)  # calculate weighted error by layer
  errors_by_time = keras.layers.Flatten()(errors_by_time)  # will be (batch_size, nt)
  final_errors = keras.layers.Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
  model = keras.models.Model(inputs=inputs, outputs=final_errors)
  model.compile(loss='mean_absolute_error', optimizer='adam')
  
  train_generator = prednet.data_utils.SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
  val_generator = prednet.data_utils.SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)
  
  lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
  callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule)]
  if save_model:
      if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
      callbacks.append(keras.callbacks.ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))
  
  history = model.fit_generator(train_generator, samples_per_epoch / batch_size, number_of_epochs, callbacks=callbacks,
                  validation_data=val_generator, validation_steps=N_seq_val / batch_size)
  
  if save_model:
      json_string = model.to_json()
      with open(json_file, "w") as f:
          f.write(json_string)
