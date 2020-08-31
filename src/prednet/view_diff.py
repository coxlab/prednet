import typing

import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt


# https://github.com/stoyanovgeorge/ffmpeg/wiki/How-to-Compare-Video
# https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
# http://www.blog.pythonlibrary.org/2016/10/11/how-to-create-a-diff-of-an-image-in-python/
#     diff = ImageChops.difference(image_one, image_two)
# https://stackoverflow.com/questions/28935851/how-to-compare-a-video-with-a-reference-video-using-opencv-and-python/30507468
# https://stackoverflow.com/questions/25774996/how-to-compare-show-the-difference-between-2-videos-in-ffmpeg
# http://www.scikit-video.org/stable/measure.html
# Full-Reference Quality Assessment
# Use skvideo.measure.scenedet to find frames that start scene boundaries. Check the documentation on selecting the specific detection algorithm.


def fig2data_alt(figure):
    # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    return np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)


# If possible, we would probably prefer to encapsulate matplotlib away from the user, in case we find a better way.
# That means no raw access to axes.
def make_comparison_image(image: np.ndarray, referenceImage: np.ndarray,
                          differenceImgFunc: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
                          numericalDifferenceVector: np.ndarray = None,
                          frameIndex: int = None,
                          showReferenceImage: bool = False) -> np.ndarray:
    """Constructs a faceted composite image showing an image alongside its diff from a reference image.
    This image is intended to be a single frame of a video; if available, a plot of the differences in over time is shown below, with the current frame labeled.
    Parameters
    ----------
    image : ndarray
        Image to display, ndarray of dimension (M, N, C), or (M, N),
        where M is the height, N is width,
        and C is number of channels.
    referenceImage : ndarray
        Reference image, ndarray of dimension (M, N, C), or (M, N),
        where M is the height, N is width,
        and C is number of channels.
    differenceImgFunc: Callable
        Function that takes two images and returns an image showing their differences.
    numericalDifferenceVector : ndarray
        Single number measuring the difference for each frame of a video, ndarray of dimension (T,),
        where T is the number of frames.
        This does not make sense for all measurements, such as ST-RRED.
    frameIndex: int
        The index of this frame in the video, if applicable.
    showReferenceImage: bool
        Whether the original reference image should be displayed.
    Returns
    -------
    composite_array : ndarray
        The comparison frame, ndarray of dimension (M, N, C) or (M, N),
        where M is the height, N is width,
        and C is number of channels.
        If the current frame is highlighted with a red dot, then C will be 3 even if the original images are grayscale.
    """
    # https://matplotlib.org/tutorials/intermediate/gridspec.html
    figure = plt.figure(constrained_layout=True)
    gridspec = figure.add_gridspec(1 if numericalDifferenceVector is None else 2, 3 if showReferenceImage else 2)
    upperLeftAx = figure.add_subplot(gridspec[0,0])
    upperLeftAx.margins(0)
    # upperLeftAx.set_title('pristine')
    upperLeftAx.imshow(image)
    if showReferenceImage:
        upperRightAx = figure.add_subplot(gridspec[0,-1])
        upperRightAx.margins(0)
        # upperRightAx.set_title('distorted')
        upperRightAx.imshow(referenceImage)
    upperMiddleAx = figure.add_subplot(gridspec[0,1])
    upperMiddleAx.margins(0)
    # upperMiddleAx.set_title('difference')
    upperMiddleAx.imshow(differenceImgFunc(image, referenceImage))
    for ax in (upperLeftAx, upperMiddleAx):
        # https://matplotlib.org/api/axes_api.html#ticks-and-tick-labels
        ax.tick_params(axis='x',  # changes apply to the x-axis
            which='both',         # both major and minor ticks are affected
            bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y',  # changes apply to the x-axis
            which='both',         # both major and minor ticks are affected
            left=False, right=False, labelleft=False)
    if numericalDifferenceVector is not None:
        bottomAx = figure.add_subplot(gridspec[1,:])
        bottomAx.margins(0)
        # bottomAx.set_title('difference')
        bottomAx.plot(numericalDifferenceVector)
        if frameIndex is not None:
            bottomAx.plot(frameIndex, numericalDifferenceVector[frameIndex], 'ro')
        bottomAx.tick_params(axis='x',  # changes apply to the x-axis
            which='both',               # both major and minor ticks are affected
            bottom=False, top=False, labelbottom=False)
        # we might want to compare anomalousness across videos
        # bottomAx.tick_params(axis='y',  # changes apply to the x-axis
        #     which='both',               # both major and minor ticks are affected
        #     left=False, right=False, labelleft=False)
    data = fig2data_alt(figure)
    plt.close(figure)
    return data


def combined_frame_shape(image: np.ndarray, referenceImage: np.ndarray,
                         differenceImgFunc: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
                         numericalDifferenceVector: np.ndarray = None,
                         frameIndex: int = None,
                         showReferenceImage: bool = False) -> typing.Tuple[int]:
    """Calculates the final size of a frame that :func:`skvideo.measure.view_diff.make_comparison_image` will make.
    There's probably a better way to do this, but for now this just actually constructs a single frame and gets its shape.
    Parameters
    ----------
    image : ndarray
        Image to display, ndarray of dimension (M, N, C), or (M, N),
        where M is the height, N is width,
        and C is number of channels.
    referenceImage : ndarray
        Reference image, ndarray of dimension (M, N, C), or (M, N),
        where M is the height, N is width,
        and C is number of channels.
    differenceImgFunc: Callable
        Function that takes two images and returns an image showing their differences.
    numericalDifferenceVector : ndarray
        Single number measuring the difference for each frame of a video, ndarray of dimension (T,),
        where T is the number of frames.
        This does not make sense for all measurements, such as ST-RRED.
    frameIndex: int
        The index of this frame in the video, if applicable.
    showReferenceImage: bool
        Whether the original reference image should be displayed.
    Returns
    -------
    shape : Tuple[int]
        dimension (M, N, C) or (M, N),
        where M is the height, N is width,
        and C is number of channels.
        If the current frame is highlighted with a red dot, then C will be 3 even if the original images are grayscale.
    """
    return make_comparison_image(image, referenceImage, differenceImgFunc, numericalDifferenceVector, frameIndex, showReferenceImage).shape


def make_comparison_video(video: np.ndarray, referenceVideo: np.ndarray,
                          differenceImgFunc: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
                          numericalDifferenceFunc: typing.Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                          showReferenceImage: bool = False) -> np.ndarray:
    """Constructs a faceted composite video showing an video alongside its diff from a reference video.
    If available, a plot of the differences in over time is shown below, with the current frame labeled.
    Parameters
    ----------
    video : ndarray
        Video to display, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.
    referenceVideo : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.
    differenceImgFunc: Callable
        Function that takes two images and returns an image showing their differences.
    numericalDifferenceVector : ndarray
        Single number measuring the difference for each frame of a video, ndarray of dimension (T,),
        where T is the number of frames.
        This does not make sense for all measurements, such as ST-RRED.
    showReferenceImage: bool
        Whether the original reference image should be displayed.
    Returns
    -------
    composite_array : ndarray
        The comparison video, (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.
        If the current frame is highlighted with a red dot, then C will be 3 even if the original images are grayscale.
    """
    assert video.shape == referenceVideo.shape
    assert video.dtype == referenceVideo.dtype
    for i in range(2):
      # The first few predicted frames are sometimes oddly dark.
      # Even 28,000 versus 35,000 can completely dominate the mean squared error.
      # is some lookback window in PredNet getting initialized to zeros when it should be copies of the first frame?
      if npla.norm(referenceVideo[0]) < npla.norm(video[0])/2 or (npla.norm(video[0]) - npla.norm(referenceVideo[0]) > (npla.norm(video[1]) - npla.norm(referenceVideo[1]))*2):
        video = video[1:]
        referenceVideo = referenceVideo[1:]
      else:
        break
    differenceVector = numericalDifferenceFunc(referenceVideo, video)
    assert len(differenceVector.shape) == 1
    assert differenceVector.shape[0] == video.shape[0]
    numberOfFrames = video.shape[0]
    compositeVideo = np.empty((numberOfFrames,) + combined_frame_shape(video[0], referenceVideo[0], differenceImgFunc, differenceVector, 0), dtype=video.dtype)
    for frameIndex, (pristineFrame, distortedFrame) in enumerate(zip(referenceVideo, video)):
        comparisonImage = make_comparison_image(distortedFrame, pristineFrame, differenceImgFunc, differenceVector, frameIndex)
        assert comparisonImage.dtype == video.dtype
        compositeVideo[frameIndex, :] = comparisonImage
    return compositeVideo


def make_prediction_comparison_video(video: np.ndarray,
                                     nextFramePredictor: typing.Callable[[np.ndarray], np.ndarray],
                                     differenceImgFunc: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     numericalDifferenceFunc: typing.Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                                    ) -> np.ndarray:
    """Constructs a faceted composite video showing an video alongside its diff from expected based on a predictor.
    Parameters
    ----------
    video : ndarray
        Video to display, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.
    nextFramePredictor : Callable
        The next-frame-predictor is taken as a video-to-video function,
        because in general prediction might depend on a sequence of frames.
    differenceImgFunc: Callable
        Function that takes two images and returns an image showing their differences.
    numericalDifferenceVector : ndarray
        Single number measuring the difference for each frame of a video, ndarray of dimension (T,),
        where T is the number of frames.
        This does not make sense for all measurements, such as ST-RRED.
    Returns
    -------
    composite_array : ndarray
        The comparison video, (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.
        If the current frame is highlighted with a red dot, then C will be 3 even if the original images are grayscale.
    """
    predictionForNextFrameAfterEach = nextFramePredictor(video)
    assert predictionForNextFrameAfterEach.shape == video.shape
    return make_comparison_video(video[1:], predictionForNextFrameAfterEach[:-1],
                                 differenceImgFunc=differenceImgFunc, numericalDifferenceFunc=numericalDifferenceFunc,
                                 showReferenceImage=False)

