import skvideo.io
import hickle
import os.path


def save_array_as_hickle(array, source_list, dirToSaveHKL):
    assert len(array.shape) == 4
    assert array.shape[3] == 3
    numberOfFrames = array.shape[0]
    trainSlice = slice(numberOfFrames//2)
    validationSlice = slice(numberOfFrames//2, numberOfFrames*3//4)
    testSlice = slice(numberOfFrames*3//4, numberOfFrames)
    train = array[trainSlice]
    validate = array[validationSlice]
    test = array[testSlice]
    hickle.dump(train, os.path.join(dirToSaveHKL, 'X_train.hkl'))
    hickle.dump(validate, os.path.join(dirToSaveHKL, 'X_validate.hkl'))
    hickle.dump(test, os.path.join(dirToSaveHKL, 'X_test.hkl'))
    # The first few sources in the KITTI sources_test.hkl are
    # 'city-2011_09_26_drive_0104_sync', 'city-2011_09_26_drive_0104_sync', 'city-2011_09_26_drive_0104_sync'
    for split, slc in (('train', trainSlice), ('validate', validationSlice), ('test', testSlice)):
        hickle.dump(array[slc], os.path.join(dirToSaveHKL, 'X_{}.hkl'.format(split)))
        hickle.dump(source_list[slc], os.path.join(dirToSaveHKL, 'sources_{}.hkl'.format(split)))


def load_video(filepath, dirToSaveHKL):
    array = skvideo.io.vread(filepath)
    # error can occur: Cannot find installation of real FFmpeg (which comes with ffprobe)
    source_list = [filepath for frame in array]
    save_array_as_hickle(array, source_list, dirToSaveHKL)

