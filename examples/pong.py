
import prednet.data_input
import prednet.train
import prednet.evaluate
import pkg_resources
import os.path
import pytest
import tempfile
import numpy as np
import hickle
import skvideo.io


def moving_dot():
  filepath = 'dot-moving-left-to-right.mpg'
  rightToLeftFilepath = 'dot-moving-right-to-left.mpg'
  leftToRight = np.zeros((2**20, 8, 8, 3), dtype=np.uint8)
  for i in range(32):
    leftToRight[i, 4, i % 8, :] = 255
  skvideo.io.vwrite(filepath, leftToRight)
  prednet.train.train_on_single_video(filepath)
  predicted = prednet.evaluate.save_predicted_frames_for_single_video(filepath,
            nt=None,
            model_file_path=prednet.train.default_path_to_save_model(filepath),
            )
  if predicted.shape != leftToRight.shape:
    raise Exception(predicted.shape)
  assert np.count_nonzero(predicted) > 0
  rightToLeft = np.zeros((32, 8, 8, 3), dtype=np.uint8)
  for i in range(32):
    rightToLeft[i, 4, -i % 8, :] = 255
  skvideo.io.vwrite(rightToLeftFilepath, rightToLeft)
  rightToLeftPredicted = prednet.evaluate.save_predicted_frames_for_single_video(rightToLeftFilepath,
            nt=None,
            model_file_path=prednet.train.default_path_to_save_model(filepath))
  assert rightToLeftPredicted.shape == rightToLeft.shape
  assert np.count_nonzero(rightToLeftPredicted) > 0

  predicted = skvideo.io.vread(prednet.evaluate.default_prediction_filepath(filepath))
  if predicted.shape != leftToRight.shape:
    raise Exception(predicted.shape)
  assert np.count_nonzero(predicted) == 0 # ???
  # maybe the reduced precision reduces near-zeros to zero

  assert np.mean( (rightToLeftPredicted[:-1] - rightToLeft[1:])**2 ) > np.mean( (predicted[:-1] - leftToRight[1:])**2 )


if __name__ == "__main__":
  moving_dot()


