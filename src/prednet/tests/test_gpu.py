
import os
import pytest

@pytest.mark.skipif('PREDNET_TESTS_INSIST_GPU' not in os.environ, reason='The test suite might be run on a machine that truly does not have a GPU, so this test is only run if specifically asked for.')
def test_gpu():
  """
  This is primarily useful for testing that the Docker image is properly built to access a GPU,
  but it can also be used in general when you want to be sure the tests are passing on the GPU
  (rather than passing but actually running on CPU).
  """
  import tensorflow
  assert tensorflow.test.is_built_with_cuda()
  assert tensorflow.test.is_gpu_available()


