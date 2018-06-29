'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import requests
from bs4 import BeautifulSoup
import urllib
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from kitti_settings import *


desired_im_sz = (128, 160)
categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Download raw zip files by scraping KITTI website
def download_data():
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    for c in categories:
        url = "http://www.cvlibs.net/datasets/kitti/raw_data.php?type=" + c
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        drive_list = soup.find_all("h3")
        drive_list = [d.text[:d.text.find(' ')] for d in drive_list]
        print "Downloading set: " + c
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir): os.mkdir(c_dir)
        for i, d in enumerate(drive_list):
            print str(i+1) + '/' + str(len(drive_list)) + ": " + d
            url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/" + d + "/" + d + "_sync.zip"
            urllib.urlretrieve(url, filename=c_dir + d + "_sync.zip")


# unzip images
def extract_data():
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw/', c + '/')
        _, _, zip_files = os.walk(c_dir).next()
        for f in zip_files:
            print 'unpacking: ' + f
            spec_folder = f[:10] + '/' + f[:-4] + '/image_03/data*'
            command = 'unzip -qq ' + c_dir + f + ' ' + spec_folder + ' -d ' + c_dir + f[:-4]
            os.system(command)


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        _, folders, _ = os.walk(c_dir).next()
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/')
            _, _, files = os.walk(im_dir).next()
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images'
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    download_data()
    extract_data()
    process_data()
