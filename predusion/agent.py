# A wrapper to PredNet using better interface
import os
import numpy as np

from prednet import PredNet
from keras.models import Model, model_from_json
from keras import backend as K
from keras.layers import Input

class Agent():

    def read_from_json(self, json_file, weights_file):
        '''
        json_file (str): typically would be os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
        weights_file (str): typically would be os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
        nt (int): number of images in a sequence
        '''
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()
        self.train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
        self.train_model.load_weights(weights_file)

    def _build_test_prednet(self, output_mode):
        # Create testing model (to output predictions)
        layer_config = self.train_model.layers[1].get_config()
        layer_config['output_mode'] = output_mode
        data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        test_prednet = PredNet(weights=self.train_model.layers[1].get_weights(), **layer_config)

        return test_prednet

    def output(self, seq, cha_first=False, is_upscaled=True, output_mode='prediction'):
        '''
        input:
          seq (np array, numeric, [number of images in each sequence, imshape[0], imshape[1], 3 channels]): if cha_first is false, the final three dimension should be imshape[0], imshape[1], 3 channels; if cha_first is ture, the final three dimensions should be 3 channels, imshape[0], imshape[1].
          is_upscaled (bool): True means the RGB value in the seq ranges from 0 to 255 and need to be normalized. The output seq_hat RGB values are in the same range as the input seq.
        '''
        test_prednet = self._build_test_prednet(output_mode)

        input_shape = list(self.train_model.layers[0].batch_input_shape[1:]) # find the input shape, (number of images, 3 channels, imshape[0], imshape[1]) if the channel_first = True
        input_shape[0] = seq.shape[0]
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)
        test_model = Model(inputs=inputs, outputs=predictions)

        seq_wrapper = seq[None, ...] # the first dimension is the number of sequence which is one

        if is_upscaled:
            seq_wrapper = seq_wrapper / 255

        if cha_first:
            seq_hat = test_model.predict(seq_wrapper, batch_size=1)
        else:
            seq_tran = np.transpose(seq_wrapper, (0, 1, 4, 2, 3)) # make it channel first
            seq_hat = test_model.predict(seq_tran, batch_size=1)
            seq_hat = np.transpose(seq_hat, (0, 1, 3, 4, 2)) # convert to original shape

        if is_upscaled:
            seq_hat = seq_hat * 255

        return seq_hat[0]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import predusion.immaker as immaker
    from kitti_settings import *

    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

    imshape = (128, 160)
    square = immaker.Square(imshape, background=150)
    seq_gener = immaker.Seq_gen()

    im = square.set_full_square(color=[255, 0, 0])
    seq_repeat = seq_gener.repeat(im, 4)

    sub = Agent()
    sub.read_from_json(json_file, weights_file)

    ##### show the prediction
    seq_pred = sub.output(seq_repeat)
    f, ax = plt.subplots(2, 3, sharey=True, sharex=True)
    for i, sq_p, sq_r in zip(range(3), seq_pred, seq_repeat):
        ax[0][i].imshow(sq_r.astype(int))
        ax[1][i].imshow(sq_p.astype(int))

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.show()

    ##### show the R2 neural activity
    r2 = sub.output(seq_repeat, output_mode='R2') # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)
    print(r2.shape)
