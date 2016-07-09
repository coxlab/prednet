import numpy as np

from keras import backend as K
from keras import activations
from keras.layers.recurrent import Recurrent
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D
from keras.engine.topology import InputSpec


class PredNet(Recurrent):
    '''PredNet architecture - Lotter 2016.
        Stacked convolutional LSTM inspired by predictive coding principles.

    # Arguments
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
        A_filt_sizes: filter sizes for the target (A) modules.
            Has length of 1 - len(stack_sizes).
            Ex. (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of the errors (E)
                from the layer below (followed by max-pooling)
        A_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', or 'all'.
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # References
        - [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        - [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
        - [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
    '''
    def __init__(self, stack_sizes, R_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error',
                 dim_ordering=K.image_dim_ordering(), **kwargs):
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max
        self.error_activation = activations.get(error_activation)
        self.LSTM_activation = activations.get(LSTM_activation)
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation)

        assert output_mode in {'prediction', 'error', 'all'}, 'output_mode must be in {prediction, error, all}'
        self.output_mode = output_mode

        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.channel_axis = -3 if dim_ordering == 'th' else -1

        super(PredNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5)]

    def get_output_shape_for(self, input_shape):
        if self.output_mode == 'prediction':
            if self.return_sequences:
                return input_shape
            else:
                return (input_shape[0],) + input_shape[2:]
        elif self.output_mode == 'error':
            if self.return_sequences:
                return (input_shape[0], input_shape[1], self.nb_layers)
            else:
                return (input_shape[0], self.nb_layers)
        else:
            if self.return_sequences:
                return (input_shape[0], input_shape[1], np.prod(input_shape[2:]) + self.nb_layers)
            else:
                return (input_shape[0], np.prod(input_shape[2:]) + self.nb_layers)


    def get_initial_states(self, x):
        input_shape = self.input_spec[0].shape
        initial_states = []
        if self.dim_ordering == 'th':
            init_nb_row = input_shape[3]
            init_nb_col = input_shape[4]
        else:
            init_nb_row = input_shape[2]
            init_nb_col = input_shape[3]

        base_initial_state = K.zeros_like(x)
        for _ in range(2):
            if self.dim_ordering == 'th':
                base_initial_state = K.sum(base_initial_state, axis=-1)
            else:
                base_initial_state = K.sum(base_initial_state, axis=-2)
        base_initial_state = K.sum(base_initial_state, axis=1)

        for u in ['r', 'c', 'e']:
            for l in range(self.nb_layers):
                ds_factor = 2 ** l
                if u in ['r', 'c']:
                    output_dim = self.R_stack_sizes[l] * (init_nb_row // ds_factor) * (init_nb_col // ds_factor)
                    if self.dim_ordering == 'th':
                        output_shp = (-1, self.R_stack_sizes[l], init_nb_row // ds_factor, init_nb_col // ds_factor)
                    else:
                        output_shp = (-1, init_nb_row // ds_factor, init_nb_col // ds_factor, self.R_stack_sizes[l])
                else:
                    output_dim = 2 * self.stack_sizes[l] * (init_nb_row // ds_factor) * (init_nb_col // ds_factor)
                    if self.dim_ordering == 'th':
                        output_shp = (-1, 2 * self.stack_sizes[l], init_nb_row // ds_factor, init_nb_col // ds_factor)
                    else:
                        output_shp = (-1, init_nb_row // ds_factor, init_nb_col // ds_factor, 2 * self.stack_sizes[l])


                reducer = K.zeros((input_shape[self.channel_axis], output_dim))

                initial_state = K.dot(base_initial_state, reducer) # samples, output_dim
                initial_state = K.reshape(initial_state, output_shp)
                initial_states += [initial_state]

        if K._BACKEND == 'theano':
            from theano import tensor as T
            # There is a known issue in the Theano scan op when dealing with inputs whose shape is 1 along a dimension.
            # In our case, this is a problem when training on grayscale images, and the below line fixes it.
            initial_states = [T.unbroadcast(init_state, 0, 1) for init_state in initial_states]
        return initial_states

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'a', 'ahat']}

        for l in range(self.nb_layers):
            for c in ['i', 'f', 'c', 'o']:
                act = self.LSTM_activation.__name__ if c == 'c' else self.LSTM_inner_activation.__name__
                self.conv_layers[c].append(Convolution2D(self.R_stack_sizes[l], self.R_filt_sizes[l], self.R_filt_sizes[l], border_mode='same', activation=act, dim_ordering=self.dim_ordering))

            self.conv_layers['ahat'].append(Convolution2D(self.stack_sizes[l], self.Ahat_filt_sizes[l], self.Ahat_filt_sizes[l], border_mode='same', activation='relu', dim_ordering=self.dim_ordering))

            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(Convolution2D(self.stack_sizes[l+1], self.A_filt_sizes[l], self.A_filt_sizes[l], border_mode='same', activation='relu', dim_ordering=self.dim_ordering))

        self.upsample = UpSampling2D()
        self.pool = MaxPooling2D()

        self.trainable_weights = []
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                elif c == 'a':
                    nb_channels = 2 * self.R_stack_sizes[l]
                else:
                    nb_channels = self.stack_sizes[l] * 2 + self.R_stack_sizes[l]
                    if l < self.nb_layers - 1:
                        nb_channels += self.R_stack_sizes[l+1]
                in_shape = (input_shape[0], nb_channels, input_shape[-2] // ds_factor, input_shape[-1] // ds_factor)
                self.conv_layers[c][l].build(in_shape)
                self.trainable_weights += self.conv_layers[c][l].trainable_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.states = [None] * self.nb_layers*3


    def step(self, a, states):
        r_tm1 = states[:self.nb_layers]
        c_tm1 = states[self.nb_layers:2*self.nb_layers]
        e_tm1 = states[2*self.nb_layers:]

        c = []
        r = []
        e = []

        for l in reversed(range(self.nb_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_layers - 1:
                inputs.append(r_up)

            inputs = K.concatenate(inputs, axis=self.channel_axis)
            i = self.conv_layers['i'][l].call(inputs)
            f = self.conv_layers['f'][l].call(inputs)
            o = self.conv_layers['o'][l].call(inputs)
            _c = f * c_tm1[l] + i * self.conv_layers['c'][l].call(inputs)
            _r = o * self.LSTM_activation(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)

        for l in range(self.nb_layers):
            ahat = self.conv_layers['ahat'][l].call(r[l])
            if l == 0:
                ahat = K.minimum(ahat, self.pixel_max)
                frame_prediction = ahat

            # compute errors
            e_up = self.error_activation(ahat - a)
            e_down = self.error_activation(a - ahat)

            e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))

            if l < self.nb_layers - 1:
                a = self.conv_layers['a'][l].call(e[l])
                a = self.pool.call(a)  # target for next layer

        if self.output_mode == 'prediction':
            output = frame_prediction
        else:
            for l in range(self.nb_layers):
                layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
            if self.output_mode == 'error':
                output = all_error
            else:
                output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

        states = r + c + e

        return output, states

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'A_filt_sizes': self.A_filt_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'output_mode': self.output_mode}
        base_config = super(PredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
