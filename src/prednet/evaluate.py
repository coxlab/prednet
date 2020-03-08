'''
Calculates mean-squared error and plots predictions.
'''

import os
import resource
import numpy as np
import skvideo.io
# import skvideo.measure.view_diff
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image, ImageChops

from keras import backend as K
from keras.models import Model, model_from_json
import keras.models
from keras.layers import Input, Dense, Flatten

from prednet.prednet import PredNet
from prednet.data_utils import SequenceGenerator
import prednet.train


def ImageChops_on_ndarrays(distortedFrame, pristineFrame):
    return ImageChops.difference(Image.fromarray(distortedFrame), Image.fromarray(pristineFrame))


def save_results(X_test, X_hat, nt, RESULTS_SAVE_DIR, path_to_save_prediction_scores: str = None):
  if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)

  if path_to_save_prediction_scores:
    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
    mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
    with open(os.path.join(RESULTS_SAVE_DIR, path_to_save_prediction_scores), 'w') as f:
      f.write("Model MSE: %f\n" % mse_model)
      f.write("Previous Frame MSE: %f" % mse_prev)
  
  n_plot = 40

  # Plot some predictions
  aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
  plt.figure(figsize = (nt, 2*aspect_ratio))
  gs = gridspec.GridSpec(2, nt)
  gs.update(wspace=0., hspace=0.)
  plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
  if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
  plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
  for i in plot_idx:
      for t in range(nt):
          plt.subplot(gs[t])
          plt.imshow(X_test[i,t], interpolation='none')
          plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          if t==0: plt.ylabel('Actual', fontsize=10)
  
          plt.subplot(gs[t + nt])
          plt.imshow(X_hat[i,t], interpolation='none')
          plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          if t==0: plt.ylabel('Predicted', fontsize=10)
  
      plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
      plt.clf()


def evaluate_on_hickles(DATA_DIR,
                        path_to_model_json='prednet_model.json', weights_path='prednet_weights.hdf5',
                        RESULTS_SAVE_DIR: str = None,
                        path_to_save_prediction_scores: str = None):
  test_file = os.path.join(DATA_DIR, 'X_test.hkl')
  assert os.path.exists(test_file)
  test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')
  assert os.path.exists(test_sources)
  return evaluate_json_model(test_file, test_sources, path_to_model_json, weights_path,
                             RESULTS_SAVE_DIR=RESULTS_SAVE_DIR,
                             path_to_save_prediction_scores=path_to_save_prediction_scores)


def get_predicted_frames_for_single_video(path_to_video,
                                          number_of_epochs=150, steps_per_epoch=125,
                                          nt=8,
                                          model_file_path=None,
                                          *args, **kwargs):
  path_to_save_model_json = os.path.splitext(path_to_video)[0] + '.model.json'
  path_to_save_weights_hdf5 = os.path.splitext(path_to_video)[0] + '.model.hdf5'
  if model_file_path is None:
    model_file_path = prednet.train.default_path_to_save_model(path_to_video)
  if not os.path.exists(model_file_path):
    prednet.train.train_on_single_video(path_to_video,
                                        path_to_save_model_json=path_to_save_model_json,
                                        path_to_save_weights_hdf5=path_to_save_weights_hdf5,
                                        path_to_save_model_file=model_file_path,
                                        number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch,
                                        *args, **kwargs)

  array = skvideo.io.vread(path_to_video)
  print('get_predicted_frames_for_single_video returned from skvideo.io.vread, memory usage',
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  assert array.dtype == np.uint8
  source_list = [path_to_video for frame in array]
  assert len(source_list) == array.shape[0]
  assert len(array.shape) == 4
  if nt is None:
    # Just evaluate the whole thing as one long sequence.
    nt = array.shape[0]
  lengthOfVideoSequences = nt
  # If lengthOfVideoSequences does not divide the number of frames,
  # PredNet will truncate, which is usually not what we want.
  if array.shape[0] % lengthOfVideoSequences != 0:
    array = np.pad(array,
                   ((0, lengthOfVideoSequences - (array.shape[0] % lengthOfVideoSequences)), (0,0), (0,0), (0,0)),
                   'edge')
    source_list.extend([source_list[-1]] * (lengthOfVideoSequences - (len(source_list) % lengthOfVideoSequences)))
  assert array.shape[0] % lengthOfVideoSequences == 0
  assert len(source_list) == array.shape[0]
  assert len(array.shape) == 4
  prediction = evaluate_json_model(array, source_list,
                             path_to_model_json=path_to_save_model_json,
                             weights_path=path_to_save_weights_hdf5,
                             model_file_path=model_file_path,
                             nt=nt)
  # if prediction.shape != array.shape:
  #   raise Exception(array.shape, prediction.shape)
  # Predictions are initially returned as float32, possibly because the model is float32.
  return prediction


def default_prediction_filepath(path_to_video):
  return os.path.splitext(path_to_video)[0] + '.predicted' + os.path.splitext(path_to_video)[1]


def save_video_as_images(path_to_save, frames_to_save):
  if type(frames_to_save) is not np.ndarray:
    raise ValueError(frames_to_save)
  for sequenceIndex, indexWithinSequence in np.ndindex(frames_to_save.shape[:2]):
    frame = frames_to_save[sequenceIndex, indexWithinSequence]
    assert type(frame) is np.ndarray
    assert len(frame.shape) == 3  # length, width, channels
    frame = Image.fromarray(frame)
    frame.save(path_to_save + '_' + str(sequenceIndex) + '_' + str(indexWithinSequence) + '.png')


def save_predicted_frames_for_single_video(path_to_video,
                                           number_of_epochs=150, steps_per_epoch=125,
                                           nt=8,
                                           model_file_path=None,
                                           path_to_save_predicted_frames=None,
                                           ):
  if path_to_save_predicted_frames is None:
    path_to_save_predicted_frames = default_prediction_filepath(path_to_video)
  path_to_save_comparison_video = os.path.splitext(path_to_video)[0] + '.comparison' + os.path.splitext(path_to_video)[1]
  predictedFrames = get_predicted_frames_for_single_video(path_to_video, number_of_epochs, steps_per_epoch, nt=nt,
                                                          model_file_path=model_file_path)

  assert predictedFrames.dtype == np.float32
  predictedFrames = (predictedFrames * 255).astype(np.uint8)
  assert predictedFrames.dtype == np.uint8

  if os.path.splitext(path_to_save_predicted_frames)[1] == '':
    save_video_as_images(path_to_save_predicted_frames, predictedFrames)

  # predictedFrames is as sequences, turn it into a regular video.
  assert len(predictedFrames.shape) == 5
  predictedFrames = predictedFrames.reshape(-1, *predictedFrames.shape[2:])
  assert len(predictedFrames.shape) == 4

  if '.' in path_to_save_predicted_frames:
    skvideo.io.vwrite(path_to_save_predicted_frames, predictedFrames)
  # raise Exception(path_to_save_predicted_frames, predictedFrames.shape)
  return predictedFrames

  # comparisonFrames = skvideo.measure.view_diff.make_comparison_video(skvideo.io.vread(path_to_video), predictedFrames, ImageChops_on_ndarrays, skvideo.measure.mse_rgb)
  # skvideo.io.vwrite(path_to_save_comparison_video, comparisonFrames)


def save_predicted_frames_for_video_list(paths_to_videos,
                                         model_file_path,
                                         number_of_epochs=150, steps_per_epoch=125,
                                         nt=8,
                                         prediction_save_extension=None,
                                         ):
  for path_to_video in paths_to_videos:
    if prediction_save_extension == 'png':
      directoryToSave = os.path.splitext(path_to_video)[0] + '_predicted_frames'
      if not os.path.exists(directoryToSave):
        os.mkdir(directoryToSave)
      path_to_save_predicted_frames = os.path.join(directoryToSave, 'predicted')
    else:
      path_to_save_predicted_frames = None
    save_predicted_frames_for_single_video(path_to_video, model_file_path=model_file_path,
                                           number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, nt=nt,
                                           path_to_save_predicted_frames=path_to_save_predicted_frames)


def make_evaluation_model(path_to_model_json='prednet_model.json', weights_path='prednet_weights.hdf5',
                          model_file_path=None,
                          nt=8):
  if model_file_path:
    train_model = keras.models.load_model(model_file_path, custom_objects = {'PredNet': PredNet})
  else:
    # Load the model and its trained weights.
    with open(path_to_model_json) as f:
      json_string = f.read()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    assert os.path.exists(weights_path)
    train_model.load_weights(weights_path)

  # Create testing model (to output predictions)
  layer_config = train_model.layers[1].get_config()
  layer_config['output_mode'] = 'prediction'
  test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
  input_shape = list(train_model.layers[0].batch_input_shape[1:])
  input_shape[0] = nt
  inputs = Input(shape=tuple(input_shape))
  predictions = test_prednet(inputs)
  data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
  return Model(inputs=inputs, outputs=predictions), data_format


def evaluate_json_model(test_file, test_sources,
                        path_to_model_json='prednet_model.json', weights_path='prednet_weights.hdf5',
                        model_file_path=None,
                        RESULTS_SAVE_DIR: str = None,
                        path_to_save_prediction_scores: str = None,
                        nt=8):
  batch_size = 4
  test_model, data_format = make_evaluation_model(path_to_model_json, weights_path, model_file_path, nt)

  test_generator = SequenceGenerator(test_file, test_sources, nt,
                                     sequence_start_mode='unique', data_format=data_format)
  X_test = test_generator.create_all()
  assert type(X_test) is np.ndarray
  assert len(X_test.shape) == 5
  assert X_test.shape[1] == nt
  # assert np.count_nonzero(X_test[:, nt-1, :, :, :]) == 0 # X_test does not have the inserted black frames?
  X_hat = test_model.predict(X_test, batch_size)
  if type(X_hat) is list:
    X_hat = np.array(X_hat)
  assert type(X_hat) is np.ndarray
  if X_hat.shape != X_test.shape:
    raise Exception(X_test.shape, X_hat.shape)
  if data_format == 'channels_first':
      X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
      X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

  if RESULTS_SAVE_DIR or path_to_save_prediction_scores:
    save_results(X_test, X_hat, nt, RESULTS_SAVE_DIR, path_to_save_prediction_scores)

  # They compare X_test[:, 1:] to X_hat[:, 1:]? Why?
  # Wouldn't the frame in X_hat at the same index be what's predicted for the *next* frame? Shouldn't it be compared to the next frame?
  assert X_hat.shape == X_test.shape
  assert X_hat.dtype == X_test.dtype
  # assert np.count_nonzero(X_hat[:, 0, :, :, :]) == 0
  return X_hat
