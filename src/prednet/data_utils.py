import hickle as hkl
import numpy as np
import typing

from keras import backend as K
from keras.preprocessing.image import Iterator

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    """
    `possible_starts` is an array of all indices of frames that could serve as the start of a sequence.
    If sequence_start_mode='all', then `possible_starts` is simply
    all indices such that the next few frames are from the same source.
    If sequence_start_mode='unique', then `possible_starts` is spaced out to be non-overlapping.

    If `max_num_sequences` is set, then `possible_starts` will be truncated to have
    only `max_num_sequences` starting points (and thus the SequenceGenerator will generate at most `max_num_sequences` sequences.

    `possible_starts` might be shuffled if `shuffle` is set.
    This is the main reason the video cannot be provided as an iterator to save memory.
    """
    def __init__(self, data_file,
                 source_file,
                 sequence_length,
                 batch_size=8,
                 shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all',
                 max_num_sequences=None,
                 data_format=K.image_data_format()):
        try:
            self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        except (hkl.hickle.FileError, ValueError):
            assert isinstance(data_file, np.ndarray)
            assert data_file.dtype == np.uint8
            self.X = data_file
        if self.X.shape[0] < sequence_length:
            # If sequence_length > X.shape[0], the generator will generate zero items. That is almost certainly not what the user intended.
            raise ValueError(self.X.shape[0], sequence_length)
        try:
            self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        except hkl.hickle.FileError:
            assert isinstance(source_file, list)
            self.sources = source_file
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.sequence_length + 1)
                                             if self.sources[i] == self.sources[i + self.sequence_length - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            assert curr_location < self.X.shape[0] - self.sequence_length + 1
            while curr_location < self.X.shape[0] - self.sequence_length + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.sequence_length - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.sequence_length
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if max_num_sequences is not None and len(self.possible_starts) > max_num_sequences:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:max_num_sequences]
        self.N_sequences = len(self.possible_starts)
        assert self.N_sequences > 0
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    # Something seems terribly wrong here. sequenceGenerator[idx] will ignore the index. How is this used?
    def __getitem__(self, null):
        return self.next()

    def next(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        The first array is a batch of frame sequences.
        The batch is of length `batch_size`, and each sequence has `sequence_length` frames.
        """
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            # self.index_generator is inherited from keras_preprocessing.image.iterator.Iterator
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.sequence_length) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.sequence_length])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        """ Currently this just scaled the images to [0,1]
        
        Arguments:
            X {np.array} -- array of images n_images * height * width * depth
        
        Returns:
            [type] -- [description]
        """
        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.sequence_length) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.sequence_length])
        return X_all


class TestsetGenerator(Iterator):
    """ Data generator that creates sequences that represent the original recordings
    
    Arguments:
        Iterator {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, data_file, source_file, nt=None, batch_size=8, shuffle=False, seed=None, output_mode='error', N_seq=None, data_format=K.image_data_format()):
        """ Initializer
        
        Arguments:
            data_file {[type]} -- [description]
            source_file {[type]} -- [description]
        
        Keyword Arguments:
            nt {[type]} -- [description] (default: {None})
            batch_size {int} -- [description] (default: {8})
            shuffle {bool} -- [description] (default: {False})
            seed {[type]} -- [description] (default: {None})
            output_mode {str} -- [description] (default: {'error'})
            N_seq {[type]} -- [description] (default: {None})
            data_format {[type]} -- [description] (default: {K.image_data_format()})
        """
        
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        self.possible_starts = [0]
        current_source = self.sources[0]
        for i, source in enumerate(self.sources):
            if source != current_source:
                self.possible_starts.append(i)
                current_source = source

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(TestsetGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        """ Currently this just scaled the images to [0,1]
        
        Arguments:
            X {np.array} -- array of images n_images * height * width * depth
        
        Returns:
            [type] -- [description]
        """
        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all
