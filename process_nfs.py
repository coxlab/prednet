'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from PIL import Image
import hickle as hkl
from nfs_settings import *


desired_im_sz = (128, 160)
video_folder = 'zebra_fish/'
# categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
# val_recordings = [('city', '2011_09_26_drive_0005_sync')]
# test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

# if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Create image datasets.
def process_data():

    im_list = []
    source_list = []  # corresponds to recording that image came from

    im_dir = os.path.join(DATA_DIR, video_folder)
    files = list(os.walk(im_dir, topdown=False))[-1][-1]
    im_list += [im_dir + f for f in sorted(files)]
    source_list += [video_folder] * len(files)
    print( 'Creating test data: ' + str(len(im_list)) + ' images')
    X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)

    for i, im_file in enumerate(im_list):
        im = imread(im_file)
        X[i] = process_im(im, desired_im_sz)

    hkl.dump(X, os.path.join(DATA_DIR, 'zebra_fish_test.hkl'))
    hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_zebra_fish.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    print('ran 1')
    target_ds = float(desired_sz[0])/im.shape[0]
    print('ran 2')
    im = np.array(Image.fromarray(im).resize((int(np.round(target_ds * im.shape[1])), desired_sz[0] )))
    print('ran 3')
    d = int((im.shape[1] - desired_sz[1]) / 2)
    print('ran 4')
    im = im[:, d:d+desired_sz[1]]
    print('ran 5')
    return im


if __name__ == '__main__':
    process_data()