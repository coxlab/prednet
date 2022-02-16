# functions for plotting different figures
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

class Ploter():

    @staticmethod
    def plot_seq_prediction(stimuli, prediction):
        '''
        plot prediction of one sequence
        input:
          stimuli (n_image, *imshape, 3): rgb color
          prediction (n_image, *imshape, 3): the output of Agent() while the output_mode is prediction
        output:
          fig, ax
        '''

        n_image = stimuli.shape[0]
        fig = plt.figure(figsize = (n_image, 2))
        gs = gridspec.GridSpec(2, n_image)
        gs.update(wspace=0., hspace=0.)

        for t, sq_s, sq_p in zip(range(n_image), stimuli, prediction):
            plt.subplot(gs[t])
            plt.imshow(sq_s.astype(int))
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

            plt.subplot(gs[t + n_image])
            plt.imshow(sq_p.astype(int))
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        return fig, gs

    def plot_seq(seq):
        '''
        plot a sequence
        input:
          seq (n_image, *imshape, 3): rgb color
        output:
          fig, ax
        '''

        n_image = seq.shape[0]
        fig = plt.figure(figsize = (n_image, 1))
        gs = gridspec.GridSpec(1, n_image)
        gs.update(wspace=0., hspace=0.)

        for t, sq in zip(range(n_image), seq):
            plt.subplot(gs[t])
            plt.imshow(sq.astype(np.uint8))
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        return fig, gs

##### Plot out colorbar
#import matplotlib.gridspec as gridspec
#
#imshape = (160 * 5, 160)
#
#square = immaker.Square(imshape, background=bg)
#
#seq_list = []
#for color in color_list:
#    r, g, b = color
#    im = square.set_full_square(color=color)
#    seq_list.append(im)
#
#seq_list = np.array(seq_list)
#n_color = color_list.shape[0]
#
#plt.figure(figsize = (n_color, imshape[0]/imshape[1]))
#gs = gridspec.GridSpec(1, n_color)
#gs.update(wspace=0., hspace=0.)
#
#for t, sq in zip(range(n_color), seq_list):
#    plt.subplot(gs[t])
#    plt.imshow(sq.astype(int))
#    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
#plt.savefig(plot_save_dir + 'xbar'  + '.png')
#plt.show()

if __name__ == '__main__':
    import os
    import predusion.immaker as immaker
    from predusion.immaker import Seq_gen
    from kitti_settings import *
    from predusion.agent import Agent

    ##### load the prednet
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

    sub = Agent()
    sub.read_from_json(json_file, weights_file)

    ###### generate images
    imshape = (128, 160)
    square = immaker.Square(imshape)

    im = square.set_full_square(color=[0, 0, 100])
    seq_repeat = Seq_gen().repeat(im, 5)[None, ...] # add a new axis to show there's only one squence

    ##### prediction
    seq_pred = sub.output(seq_repeat)
    fig, gs = Ploter().plot_seq_prediction(seq_repeat[0], seq_pred[0])
    plt.show()
