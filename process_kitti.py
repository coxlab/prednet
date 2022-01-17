'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
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
    '''
    after downloading, the data file will be stored as the following structure
    - DATA_DIR/raw/
      - city/
        - 2011_09_26_drive_0005_sync.zip
        - 2011_09_27_drive_0005_sync.zip
      - residential/
      - road/
    '''
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    for c in categories:
        url = "http://www.cvlibs.net/datasets/kitti/raw_data.php?type=" + c
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        drive_list = soup.find_all("h3")
        drive_list = [d.text[:d.text.find(' ')] for d in drive_list]
        print( "Downloading set: " + c)
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir): os.mkdir(c_dir)
        for i, d in enumerate(drive_list):
            print( str(i+1) + '/' + str(len(drive_list)) + ": " + d)
            url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/" + d + "/" + d + "_sync.zip"
            urllib.request.urlretrieve(url, filename=c_dir + d + "_sync.zip")


# unzip images
def extract_data():
    '''
    after extracting, the file structure becomes
    - DATA_DIR/raw/
      - city/
        - 2011_09_26_drive_0005_sync.zip
        - 2011_09_27_drive_0005_sync.zip
        - 2011_09_26_drive_0005_sync/
          - 2011_09_26/
            - 2011_09_26_drive_0005_sync/
              - image_03/
                - data/
      - residential/
      - road/
    '''
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw/', c + '/')
        zip_files = list(os.walk(c_dir, topdown=False))[-1][-1]#.next()
        for f in zip_files:
            print( 'unpacking: ' + f)
            spec_folder = f[:10] + '/' + f[:-4] + '/image_03/data*' # 2011_09_27/2011_09_27_drive_0005_sync/image_03/data*
            command = 'unzip -qq ' + c_dir + f + ' ' + spec_folder + ' -d ' + c_dir + f[:-4] # for example: unzip -qq DATA_DIR/raw/city/2011_09_27_drive_0005_sync.zip 2011_09_27/2011_09_27_drive_0005_sync/image_03/data* -d DATA_DIR/raw/city/2011_09_27_drive_0005_sync  The format is: unzip -q source_zip output_file_name -d destination_directory
            os.system(command)


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']} # splits = {'train': [], 'test': [], 'val': []}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/') # DATA_DIR/raw/city/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data/
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files) # for example source[i] = ['city-2011_09_26_drive_0005_sync', 'city-2011_09_26_drive_0005_sync', ...] the length is the number of images in the corresponding zip file

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

val_recordings = [('hermann', 'id0')]
test_recordings = [('hermann', 'id1')]
categories = ['hermann']

def my_process_data():
    '''
    this is same as process_data but I chanaged the fold to test if I can use my own images
    '''

    splits = {s: [] for s in ['train', 'test', 'val']} # splits = {'train': [], 'test': [], 'val': []}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder + '/') # DATA_DIR/raw/hermann/id2/
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files) # for example source[i] = ['hermann-id2', 'hermann-id2', ...] the length is the number of images in the corresponding zip file

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            print(im_file, im.shape)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'my_X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'my_sources_' + split + '.hkl'))

# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    #download_data()
    #extract_data()
    #process_data()
    my_process_data()
