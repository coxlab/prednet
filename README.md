# prednet

Code accompanying [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning] (https://arxiv.org/abs/1605.08104) by Bill Lotter, Gabriel Kreiman, and David Cox.

The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005).

The architecture is implemented as a custom layer in [Keras] (http://keras.io/). It is compatible with both [theano] (http://deeplearning.net/software/theano/) and [tensorflow] (https://www.tensorflow.org/) backends.
See http://keras.io/ for instructions on installing Keras and its list of dependencies.

## KITTI Demo

Code is included for training the PredNet on the raw [KITTI] (http://www.cvlibs.net/datasets/kitti/) dataset.
We include code for downloading and processing the data, as well as training and evaluating the model.
The preprocessed data can also be found [here] and the trained weights can be found [here].

### Steps
1. **Download/process data**
	```bash
	python process_kitti.py
	```
	This will scrape the KITTI website to download the raw data from the city, residential, and road categories (~165 GB) and then process the imagse (cropping, downsampling).
	Alternatively, the processed data (~3 GB) can be directly downloaded [here] and this step can be skipped.
	If directly downloaded, ... TODO:  make better process for filepath
	<br>

2. **Train model**
	```bash
	python kitti_train.py
	```
	See [Keras FAQ] (http://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu) on how to run using a GPU.
	Alternatively, pretrained weights can be directly downloaded [here].
	TODO:  say about downloading config and changing filepath
	<br>

3. **Evaluate model**
	```bash
	python kitti_evaluate.py
	```
	This will output the mean-squared error for predictions as well as making prediction plots.
