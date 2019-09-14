import os
import numpy as np

from keras import backend as K
from keras.legacy.interfaces import generate_legacy_interface, recurrent_args_preprocessor
from keras.models import model_from_json

legacy_prednet_support = generate_legacy_interface(
    allowed_positional_args=['stack_sizes', 'R_stack_sizes',
                            'A_filt_sizes', 'Ahat_filt_sizes', 'R_filt_sizes'],
    conversions=[('dim_ordering', 'data_format'),
                 ('consume_less', 'implementation')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None},
                        'consume_less': {'cpu': 0,
                                        'mem': 1,
                                        'gpu': 2}},
    preprocessor=recurrent_args_preprocessor)

# Convert old Keras (1.2) json models and weights to Keras 2.0
def convert_model_to_keras2(old_json_file, old_weights_file, new_json_file, new_weights_file):
    from prednet import PredNet
    # If using tensorflow, it doesn't allow you to load the old weights.
    if K.backend() != 'theano':
        os.environ['KERAS_BACKEND'] = backend
        reload(K)

    f = open(old_json_file, 'r')
    json_string = f.read()
    f.close()
    model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    model.load_weights(old_weights_file)

    weights = model.layers[1].get_weights()
    if weights[0].shape[0] == model.layers[1].stack_sizes[1]:
        for i, w in enumerate(weights):
            if w.ndim == 4:
                weights[i] = np.transpose(w, (2, 3, 1, 0))
        model.set_weights(weights)

    model.save_weights(new_weights_file)
    json_string = model.to_json()
    with open(new_json_file, "w") as f:
        f.write(json_string)


if __name__ == '__main__':
    old_dir = './model_data/'
    new_dir = './model_data_keras2/'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for w_tag in ['', '-Lall', '-extrapfinetuned']:
        m_tag = '' if w_tag == '-Lall' else w_tag
        convert_model_to_keras2(old_dir + 'prednet_kitti_model' + m_tag + '.json',
                                old_dir + 'prednet_kitti_weights' + w_tag + '.hdf5',
                                new_dir + 'prednet_kitti_model' + m_tag + '.json',
                                new_dir + 'prednet_kitti_weights' + w_tag + '.hdf5')
