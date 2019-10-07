import skvideo.io
import hickle
import os.path


def load_video(filepath, dirToSaveHKL):
    array = skvideo.io.vread(filepath)
    assert len(array.shape) == 4
    assert array.shape[3] == 3
    numberOfFrames = array.shape[0]
    train = array[:numberOfFrames//2]
    validate = array[numberOfFrames//2 : numberOfFrames*3//4]
    test = array[numberOfFrames*3//4:]
    hickle.dump(train, os.path.join(dirToSaveHKL, 'X_train.hkl'))
    hickle.dump(validate, os.path.join(dirToSaveHKL, 'X_validate.hkl'))
    hickle.dump(test, os.path.join(dirToSaveHKL, 'X_test.hkl'))
