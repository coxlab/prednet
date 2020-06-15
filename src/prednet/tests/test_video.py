
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


# Even though it's a small video, centaur_1.mpg still takes too long.
#@pytest.mark.skip
@pytest.mark.skipif(not skvideo.io.io._HAS_FFMPEG, reason='We cannot test loading a video without the video-loading library installed.')
def test_load_video():
  """
  Loading the MPG requires sudo apt-get install ffmpeg on Ubuntu.
  https://github.com/scikit-video/scikit-video/issues/98
  pip install git+https://github.com/scikit-video/scikit-video.git gets rid of the warnings.
  """
  filepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'centaur_1.mpg'))
  filepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'black.mpg'))
  prednet.evaluate.save_predicted_frames_for_single_video(filepath, number_of_epochs=4, steps_per_epoch=8)


def test_black(capsys):
  # if an argument has a default, pytest will not insert a fixture
  array = np.zeros((32, 8, 8, 3), dtype=np.uint8)
  # filepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'black.mpg'))
  # skvideo.io.vwrite(filepath, array)
  """
  # Having 16 frames instead of 32 frames causes training to hang indefinitely on Epoch 1/150:
Epoch 1/150
^CTraceback (most recent call last):
  File "src/prednet/tests/test_video.py", line 54, in <module>
    test_black()
  File "src/prednet/tests/test_video.py", line 43, in test_black
    prednet.train.train_on_hickles(tempdirpath, tempdirpath, array.shape[1], array.shape[2])
  File "/home/dahanna/prednet/src/prednet/train.py", line 68, in train_on_hickles
    validation_data=val_generator, validation_steps=N_seq_val / batch_size)
  File "/home/dahanna/anaconda3/envs/36env/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/home/dahanna/anaconda3/envs/36env/lib/python3.6/site-packages/keras/engine/training.py", line 1418, in fit_generator
    initial_epoch=initial_epoch)
  File "/home/dahanna/anaconda3/envs/36env/lib/python3.6/site-packages/keras/engine/training_generator.py", line 181, in fit_generator
    generator_output = next(output_generator)
  File "/home/dahanna/anaconda3/envs/36env/lib/python3.6/site-packages/keras/utils/data_utils.py", line 595, in get
    inputs = self.queue.get(block=True).get()
  File "/home/dahanna/anaconda3/envs/36env/lib/python3.6/queue.py", line 164, in get
    self.not_empty.wait()
  File "/home/dahanna/anaconda3/envs/36env/lib/python3.6/threading.py", line 295, in wait
    waiter.acquire()
KeyboardInterrupt
  """
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.save_array_as_hickle(array, ['black' for frame in array], tempdirpath)
    for filename in ('X_train.hkl', 'X_validate.hkl', 'X_test.hkl',
                     'sources_train.hkl', 'sources_validate.hkl', 'sources_test.hkl'):
      assert os.path.exists(os.path.join(tempdirpath, filename))
    for split in ('train', 'validate', 'test'):
      assert hickle.load(os.path.join(tempdirpath, 'X_{}.hkl'.format(split))).shape[0] == len(hickle.load(os.path.join(tempdirpath, 'sources_{}.hkl'.format(split))))
    with capsys.disabled():
      prednet.train.train_on_hickles(tempdirpath,
                                     number_of_epochs=4, steps_per_epoch=8,
                                     path_to_save_weights_hdf5=os.path.join(tempdirpath, 'zero_weights.hdf5'),
                                     path_to_save_model_json=os.path.join(tempdirpath, 'prednet_model.json'))
      weights_path = os.path.join(tempdirpath, 'zero_weights.hdf5')
      assert os.path.exists(weights_path)
      predicted = prednet.evaluate.evaluate_on_hickles(tempdirpath,
                                           path_to_save_prediction_scores=os.path.join(tempdirpath, 'prediction_scores.txt'),
                                           path_to_model_json=os.path.join(tempdirpath, 'prednet_model.json'),
                                           weights_path=weights_path,
                                           RESULTS_SAVE_DIR=tempdirpath)
      assert predicted.shape == (1, 8, 8, 8, 3)
      assert predicted.size == 8*8*8*3
      assert np.count_nonzero(predicted) == 0
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))


def test_moving_dot(capsys):
  filepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'dot-moving-left-to-right.mpg'))
  rightToLeftFilepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'dot-moving-right-to-left.mpg'))
  leftToRight = np.zeros((32, 8, 8, 3), dtype=np.uint8)
  for i in range(leftToRight.shape[0]):
    leftToRight[i, leftToRight.shape[1]//2, (i % leftToRight.shape[2]), :] = 255
  for i in range(32):
    leftToRight[i, 4, i % 8, :] = 255
  skvideo.io.vwrite(filepath, leftToRight)
  with capsys.disabled():
    prednet.train.train_on_single_video(filepath,
                                        number_of_epochs=8, steps_per_epoch=16)
  predicted = prednet.evaluate.save_predicted_frames_for_single_video(filepath,
            nt=None,
            model_file_path=prednet.train.default_path_to_save_model(filepath),
            )
  predicted = prednet.evaluate.get_predicted_frames_for_single_video(filepath,
            nt=None,
            model_file_path=prednet.train.default_path_to_save_model(filepath),
            )
  if predicted.size != 32*8*8*3:
    raise Exception(predicted.shape)
  if predicted.shape != leftToRight.shape:
    raise Exception(predicted.shape)
  assert np.count_nonzero(predicted) > 0
  # skvideo.io.vwrite('test.ogg', predicted) # if you look at it, it's basically black
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
  if predicted.size != 32*8*8*3:
    raise Exception(prednet.evaluate.default_prediction_filepath(filepath), predicted.shape)
  if predicted.shape != leftToRight.shape:
    raise Exception(predicted.shape)
  # assert np.count_nonzero(predicted) == 0 # ???
  # maybe the reduced precision reduces near-zeros to zero

  assert np.mean( (rightToLeftPredicted[:-1] - rightToLeft[1:])**2 ) >= np.mean( (predicted[:-1] - leftToRight[1:])**2 )

  """
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.save_array_as_hickle(array, ['moving' for frame in array], tempdirpath)
    with capsys.disabled():
      prednet.train.train_on_hickles(tempdirpath,
                                     number_of_epochs=4, steps_per_epoch=8,
                                     path_to_save_weights_hdf5=os.path.join(tempdirpath, 'weights.hdf5'),
                                     path_to_save_model_json=os.path.join(tempdirpath, 'prednet_model.json'))
      predicted = prednet.evaluate.evaluate_on_hickles(tempdirpath,
                                           path_to_save_prediction_scores=os.path.join(tempdirpath, 'prediction_scores.txt'),
                                           path_to_model_json=os.path.join(tempdirpath, 'prednet_model.json'),
                                           weights_path=os.path.join(tempdirpath, 'weights.hdf5'),
                                           RESULTS_SAVE_DIR=tempdirpath)
    assert predicted.shape == (1, 8, 8, 8, 3)
    assert predicted.size == 8*8*8*3
  """


class StubCapSys:
  def disabled(self):
    import contextlib
    return contextlib.suppress(*[])

if __name__ == "__main__":
  """
  If having GPU problems, try running with CUDA_VISIBLE_DEVICES= to run on CPU.
  """
  import signal
  def print_linenum(signum, frame):
    print("Currently at line", frame.f_lineno, frame.f_code.co_filename)
    import sys
    sys.exit()
  # signal.signal(signal.SIGINT, print_linenum)
  # test_load_video()
  # test_black(StubCapSys())
  test_moving_dot(StubCapSys())

