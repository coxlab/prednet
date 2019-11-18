'''
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
import skvideo.io
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet.prednet import PredNet
from prednet.data_utils import SequenceGenerator
import prednet.train


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
  return evaluate_json_model(test_file, test_sources, path_to_model_json, weights_path, RESULTS_SAVE_DIR, path_to_save_prediction_scores)


def get_predicted_frames_for_single_video(path_to_video,
                                          number_of_epochs=150, steps_per_epoch=125,
                                          ):
  path_to_save_model_json = os.path.splitext(path_to_video)[0] + '.model.json'
  path_to_save_weights_hdf5 = os.path.splitext(path_to_video)[0] + '.model.hdf5'
  prednet.train.train_on_single_video(path_to_video,
                                      path_to_save_model_json=path_to_save_model_json,
                                      path_to_save_weights_hdf5=path_to_save_weights_hdf5,
                                      number_of_epochs=number_of_epochs, steps_per_epoch=steps_per_epoch)
  array = skvideo.io.vread(path_to_video)
  assert array.dtype == np.uint8
  source_list = [path_to_video for frame in array]
  assert len(source_list) == array.shape[0]
  prediction = evaluate_json_model(array, source_list,
                             path_to_model_json=path_to_save_model_json,
                             weights_path=path_to_save_weights_hdf5)
  # if prediction.shape != array.shape:
  #   raise Exception(array.shape, prediction.shape)
  # Predictions are initially returned as float32, possibly because the model is float32.
  return prediction


def make_evaluation_model(path_to_model_json='prednet_model.json', weights_path='prednet_weights.hdf5',
                          nt=8):
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
                        RESULTS_SAVE_DIR: str = None,
                        path_to_save_prediction_scores: str = None):
  batch_size = 4
  nt = 8
  test_model, data_format = make_evaluation_model(path_to_model_json, weights_path, nt)

  test_generator = SequenceGenerator(test_file, test_sources, nt,
                                     sequence_start_mode='unique', data_format=data_format)
  X_test = test_generator.create_all()
  assert type(X_test) is np.ndarray
  X_hat = test_model.predict(X_test, batch_size)
  if type(X_hat) is list:
    X_hat = np.array(X_hat)
  assert type(X_hat) is np.ndarray
  if X_hat.shape != X_test.shape:
    raise Exception(X_test.shape, X_hat.shape)
  assert len(X_test.shape) == 5
  if data_format == 'channels_first':
      X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
      X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

  if RESULTS_SAVE_DIR or path_to_save_prediction_scores:
    save_results(X_test, X_hat, nt, RESULTS_SAVE_DIR, path_to_save_prediction_scores)

  # They compare X_test[:, 1:] to X_hat[:, 1:]? Why?
  # Wouldn't the frame in X_hat at the same index be what's predicted for the *next* frame? Shouldn't it be compared to the next frame?
  assert X_hat.shape == X_test.shape
  assert X_hat.dtype == X_test.dtype
  return X_hat
