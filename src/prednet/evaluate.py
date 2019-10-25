'''
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
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


def evaluate_json_model(DATA_DIR, WEIGHTS_DIR, RESULTS_SAVE_DIR,
                        json_file='prednet_model.json', weights_file='prednet_weights.hdf5',
                        path_to_save_prediction_scores: str = None):
  n_plot = 40
  batch_size = 10
  nt = 10
  
  test_file = os.path.join(DATA_DIR, 'X_test.hkl')
  test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')
  
  # Load trained model
  with open(os.path.join(WEIGHTS_DIR, json_file)) as f:
    json_string = f.read()
  train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
  train_model.load_weights(os.path.join(WEIGHTS_DIR, weights_file))
  
  # Create testing model (to output predictions)
  layer_config = train_model.layers[1].get_config()
  layer_config['output_mode'] = 'prediction'
  data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
  test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
  input_shape = list(train_model.layers[0].batch_input_shape[1:])
  input_shape[0] = nt
  inputs = Input(shape=tuple(input_shape))
  predictions = test_prednet(inputs)
  test_model = Model(inputs=inputs, outputs=predictions)
  
  test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
  X_test = test_generator.create_all()
  X_hat = test_model.predict(X_test, batch_size)
  if data_format == 'channels_first':
      X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
      X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
  
  if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)

  if path_to_save_prediction_scores:
    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
    mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
    with open(os.path.join(RESULTS_SAVE_DIR, path_to_save_prediction_scores), 'w') as f:
      f.write("Model MSE: %f\n" % mse_model)
      f.write("Previous Frame MSE: %f" % mse_prev)
  
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