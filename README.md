# prednet

Code and models accompanying [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning] (https://arxiv.org/abs/1605.08104) by Bill Lotter, Gabriel Kreiman, and David Cox.

The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005).
**Check out example prediction videos [here] (https://coxlab.github.io/prednet/).**

The architecture is implemented as a custom layer<sup>1</sup> in [Keras] (http://keras.io/). Tested on Keras 1.0.7 with [theano] (http://deeplearning.net/software/theano/) backend.
See http://keras.io/ for instructions on installing Keras and its list of dependencies.
For Torch implementation, see [torch-prednet] (https://github.com/e-lab/torch-prednet).
<br>

## KITTI Demo

Code is included for training the PredNet on the raw [KITTI] (http://www.cvlibs.net/datasets/kitti/) dataset.
We include code for downloading and processing the data, as well as training and evaluating the model.
The preprocessed data and can also be downloaded directly using `download_data.sh` and the **trained weights** by running `download_models.sh`.
The model download will include the original weights trained for t+1 prediction, as well as the fine-tuned weights trained to extrapolate predictions for multiple timesteps (see paper for details).

### Steps
1. **Download/process data**
	```bash
	python process_kitti.py
	```
	This will scrape the KITTI website to download the raw data from the city, residential, and road categories (~165 GB) and then process the images (cropping, downsampling).
	Alternatively, the processed data (~3 GB) can be directly downloaded by executing `download_data.sh`
	<br>
	<br>

2. **Train model**
	```bash
	python kitti_train.py
	```
	This will train a PredNet model for t+1 prediction.
	See [Keras FAQ] (http://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu) on how to run using a GPU.
	**To download pre-trained weights**, run `download_models.sh`
	<br>
	<br>

3. **Evaluate model**
	```bash
	python kitti_evaluate.py
	```
	This will output the mean-squared error for predictions as well as make plots comparing predictions to ground-truth.

<br>

<sup>1</sup> Note on implementation:  PredNet inherits from the Recurrent layer class, i.e. it has an internal state and a step function. Given the top-down then bottom-up update sequence, it must currently be implemented in Keras as essentially a 'super' layer where all layers in the PredNet are in one PredNet 'layer'. This is less than ideal, but it seems like the most efficient way as of now. We welcome suggestions if anyone thinks of a better implementation.  
