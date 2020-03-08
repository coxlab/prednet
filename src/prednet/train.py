
import os.path
import numpy as np
import skvideo.io
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
                                     path_to_save_model_json=path_to_save_model_json, path_to_save_weights_hdf5=path_to_save_weights_hdf5,
                                     number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch)


def default_path_to_save_model(path_to_video):
  return os.path.splitext(path_to_video)[0] + '.model.save.hdf5'


def train_on_single_video(path_to_video,
                          path_to_save_model_json=None, path_to_save_weights_hdf5=None,
                          path_to_save_model_file=None,
                          number_of_epochs=150, steps_per_epoch=125,
                          batch_size=4,
                          sequence_length=8,
                          number_of_validation_sequences=100,
                          ):
  """
  Picking which frames to use for validation is tricky, because when using sequence_start_mode='all',
  we require the video to have no jumps in it; we need to be able to start a sequence anywhere.
  Thus, the only place we can really scythe out a section for validation is at the beginning or the end.
  """
  if not path_to_save_model_json:
    path_to_save_model_json = os.path.splitext(path_to_video)[0] + '.model.json'
  if not path_to_save_weights_hdf5:
    path_to_save_weights_hdf5 = os.path.splitext(path_to_video)[0] + '.model.hdf5'
  if not path_to_save_model_file:
    path_to_save_model_file = default_path_to_save_model(path_to_video)
  path_to_save_settings = os.path.splitext(path_to_save_model_file)[0] + '.settings.txt'
  print('number_of_epochs =', number_of_epochs, 'steps_per_epoch =', steps_per_epoch,
        file=open(path_to_save_settings, 'w'))
  # Better to check in the evaluate function whether training is already done.
  # If the train function gets a command to re-train (possibly on a new video), then re-train.
  # if path_to_save_model_file and os.path.exists(path_to_save_model_file):
    # For this special case, do not re-train if we already have a trained model.
  #  print('train_on_single_video found', path_to_save_model_file,
  #        'so just using that instead of re-training.')
    # Later we should re-train here since we'll be training on multiple videos sequentially.
  #  return
  if os.path.exists(path_to_save_model_json) and os.path.exists(path_to_save_weights_hdf5):
    # For this special case, do not re-train if we already have a trained model.
    print('train_on_single_video found', path_to_save_model_json, 'and', path_to_save_weights_hdf5,
          'so just using those instead of re-training.')
    if path_to_save_model_file and not os.path.exists(path_to_save_model_file):
      with open(path_to_save_model_json) as f:
        json_string = f.read()
      model = keras.models.model_from_json(json_string, custom_objects = {'PredNet': prednet.prednet.PredNet})
      model.load_weights(path_to_save_weights_hdf5)
      model.save(path_to_save_model_file)
    return

  print('train_on_single_video about to call skvideo.io.vread')
  array = skvideo.io.vread(path_to_video)
  print('train_on_single_video returned from skvideo.io.vread')
  source_list = [path_to_video for frame in array]
  assert len(source_list) == array.shape[0]
  numberOfFrames = array.shape[0]
  numberOfValidationFrames = number_of_validation_sequences * sequence_length
  return train_on_arrays_and_sources(array[:-numberOfValidationFrames], source_list[:-numberOfValidationFrames],
                                     array[-numberOfValidationFrames:], source_list[-numberOfValidationFrames:],
                                     path_to_save_model_json=path_to_save_model_json,
                                     path_to_save_weights_hdf5=path_to_save_weights_hdf5,
                                     model_path=path_to_save_model_file,
                                     number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch,
                                     batch_size=batch_size, sequence_length=sequence_length,
                                     )



def train_on_single_path(path,
                         path_to_save_model_file=None,
                         number_of_epochs=150, steps_per_epoch=125,
                         *args, **kwargs):
  # os.walk() on a file name returns an empty list instead of a list containing only that entry,
  # so we need to specifically check whether the path is a directory.
  if os.path.isdir(path):
    for root, dirs, files in os.walk(path):
      for filename in files:
        # should probably check whether it's actually a video file
        train_on_single_video(os.path.join(root, filename), path_to_save_model_file=path_to_save_model_file,
                              number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch,
                              *args, **kwargs)
  else:
    train_on_single_video(path, path_to_save_model_file=path_to_save_model_file,	
                          number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch,
                          *args, **kwargs)


def train_on_video_list(paths_to_videos,
                        path_to_save_model_file,
                        number_of_epochs=150, steps_per_epoch=125,
                        *args, **kwargs):
  for path_to_video in paths_to_videos:
    train_on_single_path(path_to_video, path_to_save_model_file=path_to_save_model_file,
                         number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch,
                         *args, **kwargs)


def make_training_model(nt, input_shape):
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
  return model


def train_on_arrays_and_sources(train_file, train_sources, val_file, val_sources,
                                path_to_save_model_json='prednet_model.json', path_to_save_weights_hdf5='prednet_weights.hdf5',
                                model_path=None,
                                number_of_epochs=150, steps_per_epoch=125,
                                batch_size=4,
                                sequence_length=8,
                                max_validation_sequences=100,
                                ):
  """
  At most `max_validation_sequences` frame sequences will be used for validation.
  The validation data will be used at the end of each epoch.
  The loss function on validation data will be reported as val_loss.
  https://keras.io/models/sequential/#fit_generator
Epoch 3/8
16/16 [==============================] - 25s 2s/step - loss: 0.0064 - val_loss: 0.0064
Epoch 4/8
16/16 [==============================] - 27s 2s/step - loss: 0.0067 - val_loss: 0.0066

  `sequence_length` is the number of frames in each training sequence.
  """

  save_model = True  # if weights will be saved

  # Training parameters
  samples_per_epoch = steps_per_epoch * batch_size

  train_generator = prednet.data_utils.SequenceGenerator(train_file, train_sources,
                                                         sequence_length=sequence_length, batch_size=batch_size,
                                                         shuffle=True)
  val_generator = prednet.data_utils.SequenceGenerator(val_file, val_sources,
                                                       sequence_length=sequence_length, batch_size=batch_size,
                                                       max_num_sequences=max_validation_sequences)
  assert train_generator.im_shape == val_generator.im_shape

  if model_path and os.path.exists(model_path):
    model = keras.models.load_model(model_path, custom_objects = {'PredNet': prednet.prednet.PredNet})
  else:
    model = make_training_model(sequence_length, train_generator.im_shape)

  lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
  callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule)]
  if save_model:
      if os.path.dirname(path_to_save_weights_hdf5) != '' and not os.path.exists(os.path.dirname(path_to_save_weights_hdf5)):
        os.makedirs(os.path.dirname(path_to_save_weights_hdf5), exist_ok=True)
      print('Setting keras.callbacks.ModelCheckpoint for', path_to_save_weights_hdf5)
      callbacks.append(keras.callbacks.ModelCheckpoint(filepath=path_to_save_weights_hdf5,
                                                       monitor='val_loss', save_best_only=True))
  else:
      raise NotImplementedError("It appears that evaluation requires the HDF5 file.")
  
  history = model.fit_generator(train_generator, steps_per_epoch, number_of_epochs,
                                callbacks=callbacks,
                  validation_data=val_generator, validation_steps=max_validation_sequences / batch_size)

  if save_model:
      if model_path:
        if os.path.dirname(path_to_save_model_json) != '':
          os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
      if os.path.dirname(path_to_save_model_json) != '' and not os.path.exists(os.path.dirname(path_to_save_model_json)):
        os.makedirs(os.path.dirname(path_to_save_model_json), exist_ok=True)
      json_string = model.to_json()
      with open(path_to_save_model_json, "w") as f:
          f.write(json_string)
      model.save_weights(path_to_save_weights_hdf5)
      assert os.path.exists(path_to_save_weights_hdf5)
