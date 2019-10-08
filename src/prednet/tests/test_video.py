
import prednet.data_input
import prednet.train
import pkg_resources
import os.path
import tempfile


def test_video():
  """
  Loading the MPG requires sudo apt-get install ffmpeg on Ubuntu.
  https://github.com/scikit-video/scikit-video/issues/98
  pip install git+https://github.com/scikit-video/scikit-video.git gets rid of the warnings.
  """
  filepath = pkg_resources.resource_filename(__name__, os.path.join('resources', 'centaur_1.mpg'))
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.load_video(filepath, tempdirpath)
    prednet.train.train_on_hickles(tempdirpath, tempdirpath)
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))
