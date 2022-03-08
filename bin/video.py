# Show the video stimuli used in paper Hénaff et al. (2021) from https://osf.io/gwtcs/
import scipy.io
import matplotlib.pyplot as plt

stim_info = scipy.io.loadmat('./data/stim_info.mat')
print(stim_info.keys())

for key in stim_info:
    try:
        print(key, stim_info[key].shape)
    except:
        print(key, stim_info[key])

#print(stim_info['artificial_movie_contrast'])
print(stim_info['artificial_movie_frame'][0, 0])
#print(stim_info['artificial_movie_labels'])

plt.figure()
for im in stim_info['artificial_movie_frame'][0]:
    plt.imshow(im)
    plt.show()

stim_matrix = scipy.io.loadmat('./data/stim_matrix.mat')
print(stim_matrix.keys())

for key in stim_matrix:
    try:
        print(key, stim_matrix[key].shape)
    except:
        print(key, stim_matrix[key])
