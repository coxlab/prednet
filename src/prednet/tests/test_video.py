
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
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.load_video(filepath, tempdirpath)
    for filename in ('X_train.hkl', 'X_validate.hkl', 'X_test.hkl',
                     'sources_train.hkl', 'sources_validate.hkl', 'sources_test.hkl'):
      assert os.path.exists(os.path.join(tempdirpath, filename))
    # prednet.train.train_on_hickles(tempdirpath, tempdirpath, 240, 320)
    prednet.train.train_on_hickles(tempdirpath,
                                   number_of_epochs=4, steps_per_epoch=8,
                                   path_to_save_weights_hdf5=os.path.join(tempdirpath, 'zero_weights.hdf5'),
                                   path_to_save_model_json=os.path.join(tempdirpath, 'prednet_model.json'))
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))


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
      prednet.evaluate.evaluate_on_hickles(tempdirpath,
                                           path_to_save_prediction_scores='prediction_scores.txt',
                                           path_to_model_json=os.path.join(tempdirpath, 'prednet_model.json'),
                                           weights_path=weights_path,
                                           RESULTS_SAVE_DIR=tempdirpath)
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))


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
  test_black(StubCapSys())
