## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
## ==============================================================================
#"""A script for testing that TensorFlow is installed correctly on Windows.
#
#The script will attempt to verify your TensorFlow installation, and print
#suggestions for how to fix your installation.
#"""
#
#import ctypes
#import imp
#import sys
#
#def main():
#  try:
#    import tensorflow as tf
#    print("TensorFlow successfully installed.")
#    if tf.test.is_built_with_cuda():
#      print("The installed version of TensorFlow includes GPU support.")
#    else:
#      print("The installed version of TensorFlow does not include GPU support.")
#    sys.exit(0)
#  except ImportError:
#    print("ERROR: Failed to import the TensorFlow module.")
#
#  print("""
#WARNING! This script is no longer maintained! 
#=============================================
#
#Since TensorFlow 1.4, the self-check has been integrated with TensorFlow itself,
#and any missing DLLs will be reported when you execute the `import tensorflow`
#statement. The error messages printed below refer to TensorFlow 1.3 and earlier,
#and are inaccurate for later versions of TensorFlow.""")
#    
#  candidate_explanation = False
#
#  python_version = sys.version_info.major, sys.version_info.minor
#  print("\n- Python version is %d.%d." % python_version)
#  if not (python_version == (3, 5) or python_version == (3, 6)):
#    candidate_explanation = True
#    print("- The official distribution of TensorFlow for Windows requires "
#          "Python version 3.5 or 3.6.")
#  
#  try:
#    _, pathname, _ = imp.find_module("tensorflow")
#    print("\n- TensorFlow is installed at: %s" % pathname)
#  except ImportError:
#    candidate_explanation = False
#    print("""
#- No module named TensorFlow is installed in this Python environment. You may
#  install it using the command `pip install tensorflow`.""")
#
#  try:
#    msvcp140 = ctypes.WinDLL("msvcp140.dll")
#  except OSError:
#    candidate_explanation = True
#    print("""
#- Could not load 'msvcp140.dll'. TensorFlow requires that this DLL be
#  installed in a directory that is named in your %PATH% environment
#  variable. You may install this DLL by downloading Microsoft Visual
#  C++ 2015 Redistributable Update 3 from this URL:
#  https://www.microsoft.com/en-us/download/details.aspx?id=53587""")
#
#  try:
#    cudart64_80 = ctypes.WinDLL("cudart64_80.dll")
#  except OSError:
#    candidate_explanation = True
#    print("""
#- Could not load 'cudart64_80.dll'. The GPU version of TensorFlow
#  requires that this DLL be installed in a directory that is named in
#  your %PATH% environment variable. Download and install CUDA 8.0 from
#  this URL: https://developer.nvidia.com/cuda-toolkit""")
#
#  try:
#    nvcuda = ctypes.WinDLL("nvcuda.dll")
#  except OSError:
#    candidate_explanation = True
#    print("""
#- Could not load 'nvcuda.dll'. The GPU version of TensorFlow requires that
#  this DLL be installed in a directory that is named in your %PATH%
#  environment variable. Typically it is installed in 'C:\Windows\System32'.
#  If it is not present, ensure that you have a CUDA-capable GPU with the
#  correct driver installed.""")
#
#  cudnn5_found = False
#  try:
#    cudnn5 = ctypes.WinDLL("cudnn64_5.dll")
#    cudnn5_found = True
#  except OSError:
#    candidate_explanation = True
#    print("""
#- Could not load 'cudnn64_5.dll'. The GPU version of TensorFlow
#  requires that this DLL be installed in a directory that is named in
#  your %PATH% environment variable. Note that installing cuDNN is a
#  separate step from installing CUDA, and it is often found in a
#  different directory from the CUDA DLLs. You may install the
#  necessary DLL by downloading cuDNN 5.1 from this URL:
#  https://developer.nvidia.com/cudnn""")
#
#  cudnn6_found = False
#  try:
#    cudnn = ctypes.WinDLL("cudnn64_6.dll")
#    cudnn6_found = True
#  except OSError:
#    candidate_explanation = True
#
#  if not cudnn5_found or not cudnn6_found:
#    print()
#    if not cudnn5_found and not cudnn6_found:
#      print("- Could not find cuDNN.")
#    elif not cudnn5_found:
#      print("- Could not find cuDNN 5.1.")
#    else:
#      print("- Could not find cuDNN 6.")
#      print("""
#  The GPU version of TensorFlow requires that the correct cuDNN DLL be installed
#  in a directory that is named in your %PATH% environment variable. Note that
#  installing cuDNN is a separate step from installing CUDA, and it is often
#  found in a different directory from the CUDA DLLs. The correct version of
#  cuDNN depends on your version of TensorFlow:
#  
#  * TensorFlow 1.2.1 or earlier requires cuDNN 5.1. ('cudnn64_5.dll')
#  * TensorFlow 1.3 or later requires cuDNN 6. ('cudnn64_6.dll')
#    
#  You may install the necessary DLL by downloading cuDNN from this URL:
#  https://developer.nvidia.com/cudnn""")
#    
#  if not candidate_explanation:
#    print("""
#- All required DLLs appear to be present. Please open an issue on the
#  TensorFlow GitHub page: https://github.com/tensorflow/tensorflow/issues""")
#
#  sys.exit(-1)
#
#if __name__ == "__main__":
#  main()
import tensorflow as tf
