# Show the video stimuli used in paper Hénaff et al. (2021) from https://osf.io/gwtcs/
import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np

#stim_info = scipy.io.loadmat('./data/stim_info.mat')
#print(stim_info.keys())
#
#for key in stim_info:
#    try:
#        print(key, stim_info[key].shape)
#    except:
#        print(key, stim_info[key])
#
#print(stim_info['artificial_movie_contrast'])
#print(stim_info['artificial_movie_frame'][0, 0])
#print(stim_info['artificial_movie_labels'])

#plt.figure()
#for im in stim_info['natural_movie_contrast']:
#    plt.imshow(im)
#    plt.show()

#plt.figure()
#for im in stim_info['natural_movie_frame'][0]:
#    plt.imshow(im)
#    plt.show()

stim_matrix = scipy.io.loadmat('./data/stim_matrix.mat')
#print(stim_matrix.keys())
#
#for key in stim_matrix:
#    try:
#        print(key, stim_matrix[key].shape)
#    except:
#        print(key, stim_matrix[key])

#print(stim_matrix['image_paths'])
#print(stim_matrix['artificial_movie_labels'])
#print(stim_matrix['natural_movie_labels'])
#i_fram='02'
#for label in stim_matrix['natural_movie_labels'][0]:
#    if ('natural' in label[0]) and ('01' in label[0]) and ('1x' in label[0]):
#        print(label)

#image_path = np.array(stim_matrix['image_paths'])
#print('01' in image_path) # element compare is not implemented in this numpy version
#print(image_path.shape)
#print(type(image_path))

plt.figure()
for i_scale, im_scale in enumerate(stim_matrix['image_paths']):

    for i_type, im_type in enumerate(im_scale):

        for i_cate, im_cate in enumerate(im_type):

            for i_frame, im_frame in enumerate(im_cate):
                if ('natural01' in im_frame[0]) and ('movie03' in im_frame[0]) and ('zoom1x' in im_frame[0]):
                    plt.imshow(stim_matrix['stim_matrix'][i_scale, i_type, i_cate, :, :, i_frame])
                    plt.show()

## export all frames in stim_matrix
#path = './data/'
#plt.figure()
#for i_scale, im_scale in enumerate(stim_matrix['stim_matrix']):
#    path = os.path.join(path, 'scale' + str(i_scale))
#
#    for i_type, im_type in enumerate(im_scale):
#        path = os.path.join(path, 'type' + str(i_type))
#
#        for i_cate, im_cate in enumerate(im_type):
#            path = os.path.join(path, 'cate' + str(i_cate))
#
#            for i_frame, im_frame in enumerate(im_cate):
#                path = os.path.join(path, 'frame' + str(i_type))
