========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|




.. end-badges

Installation
============

If prednet has been uploaded to a devpi instance your pip is connected to, then you can install with::

    pip install prednet

You can always install the bleeding-edge updates with::

    pip install git+ssh://git@github.com/coxlab/prednet.git@master


prednet
=======

Code and models accompanying `Deep Predictive Coding Networks for Video
Prediction and Unsupervised Learning`_ by Bill Lotter, Gabriel Kreiman,
and David Cox.

The PredNet is a deep recurrent convolutional neural network that is
inspired by the neuroscience concept of predictive coding (Rao and
Ballard, 1999; Friston, 2005). **Check out example prediction
videos**\ `here`_\ **.**

The architecture is implemented as a custom layer :sup:`1` in `Keras`_. Code and
model data is compatible with Keras 2.0 and Python 2.7 and 3.6. The
latest version has been tested on Keras 2.2.4 with Tensorflow 1.6. For
previous versions of the code compatible with Keras 1.2.1, use fbcdc18.
To convert old PredNet model files and weights for Keras 2.0
compatibility, see ``convert_model_to_keras2`` in ``keras_utils.py``.

KITTI Demo
----------

Code is included for training the PredNet on the raw `KITTI`_ dataset.
We include code for downloading and processing the data, as well as
training and evaluating the model. The preprocessed data and can also be
downloaded directly using ``download_data.sh`` and the **trained
weights** by running ``download_models.sh``. The model download will
include the original weights trained for t+1 prediction, the fine-tuned
weights trained to extrapolate predictions for multiple timesteps, and
the "Lall" weights trained with an 0.1 loss weight on upper layers (see
paper for details).

Steps
~~~~~

1. **Download/process data**

   .. code:: bash

      python process_kitti.py

   This will scrape the KITTI website to download the raw data from the
   city, residential, and road categories (~165 GB) and then process the
   images (cropping, downsampling). Alternatively, the processed data
   (~3 GB) can be directly downloaded by executing ``download_data.sh``

2. **Train model**

   .. code:: bash

      python kitti_train.py

   This will train a PredNet model for t+1 prediction. See `Keras FAQ`_
   on how to run using a GPU. **To download pre-trained weights**, run
   ``download_models.sh``

3. **Evaluate model**

   .. code:: bash

      python kitti_evaluate.py

   This will output the mean-squared error for predictions as well as
   make plots comparing predictions to ground-truth.

Feature Extraction
~~~~~~~~~~~~~~~~~~

Extracting the intermediate features for a given layer in the PredNet
can be done using the appropriate ``output_mode`` argument. For example,
to extract the hidden state of the LSTM (the "Representation" units) in
the lowest layer, use ``output_mode = 'R0'``. More details can be found
in the PredNet docstring.

Multi-Step Prediction
~~~~~~~~~~~~~~~~~~~~~

The PredNet argument ``extrap_start_time`` can be used to force
multi-step prediction. Starting at this time step, the prediction from
the previous time step will be treated as the actual input. For example,
if the model is run on a sequence o

.. _Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning: https://arxiv.org/abs/1605.08104
.. _here: https://coxlab.github.io/prednet/
.. _Keras: http://keras.io/
.. _KITTI: http://www.cvlibs.net/datasets/kitti/
.. _Keras FAQ: http://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu

Documentation
=============


https://coxlab.github.io/prednet/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
