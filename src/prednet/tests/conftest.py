import sys
import traceback

import pytest
import tensorflow.compat.v1

@pytest.fixture(scope="session", autouse=True)
def disable_v2_behavior():
    tensorflow.compat.v1.disable_v2_behavior()

class TracePrints(object):
  def __init__(self):
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

class TraceErr(object):
  def __init__(self):
    self.stderr = sys.stderr
  def write(self, s):
    self.stderr.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stderr)

@pytest.fixture(scope="session", autouse=True)
def trace_prints():
    sys.stdout = TracePrints()
    sys.stderr = TraceErr()

