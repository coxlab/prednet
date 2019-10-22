
import prednet.data_input
import prednet.train
import pkg_resources
import os.path
import pytest
import tempfile
import numpy as np
import hickle


# Even though it's a small video, this still takes too long.
@pytest.mark.skip
def test_video():
  """
  Loading the MPG requires sudo apt-get install ffmpeg on Ubuntu.
  https://github.com/scikit-video/scikit-video/issues/98
  pip install git+https://github.com/scikit-video/scikit-video.git gets rid of the warnings.
  """
  filepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'centaur_1.mpg'))
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.load_video(filepath, tempdirpath)
    for filename in ('X_train.hkl', 'X_validate.hkl', 'X_test.hkl',
                     'sources_train.hkl', 'sources_validate.hkl', 'sources_test.hkl'):
      assert os.path.exists(os.path.join(tempdirpath, filename))
    prednet.train.train_on_hickles(tempdirpath, tempdirpath, 240, 320)
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))


def test_black():
  array = np.zeros((8, 8, 8, 3))
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.save_array_as_hickle(array, ['black' for frame in array], tempdirpath)
    for filename in ('X_train.hkl', 'X_validate.hkl', 'X_test.hkl',
                     'sources_train.hkl', 'sources_validate.hkl', 'sources_test.hkl'):
      assert os.path.exists(os.path.join(tempdirpath, filename))
    for split in ('train', 'validate', 'test'):
      assert hickle.load(os.path.join(tempdirpath, 'X_{}.hkl'.format(split))).shape[0] == len(hickle.load(os.path.join(tempdirpath, 'sources_{}.hkl'.format(split))))
    prednet.train.train_on_hickles(tempdirpath, tempdirpath, 8, 8)
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))

