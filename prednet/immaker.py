'''
Create a series of static images to the target folder
'''
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imwrite


class immaker():
    pass

if __name__ == '__main__':
    imshape = (128, 160)
    im = np.ones(imshape, dtype=int) * 255
    offset = (0, 0) # the left upper coner of nonzero pixel
    frac_d = 16 # the length of square is equal to imashape[0] / frac_d
    ratio_db = 1.8
    n_image = 5

    gvalue = 0

    d = imshape[0] // frac_d # number of pixels in each square
    b = int(ratio_db * d) # spacing of two adjacent squares, this must be larger than d

    for idy in range(offset[0], imshape[0], b):
        for idx in range(offset[1], imshape[1], b):
            try:
                im[idy:idy+d, idx:idx+d] = gvalue
            except:
                idy_end = min(idy+d, offset[0])
                idx_end = min(idx+d, offset[1])
                im[idy:idy_end, idx:idx_end] = gvalue
    plt.figure()
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()

    for i in range(n_image):
        save_path = './figs/grid_{0}_{1}_{2}.png'.format(frac_d, ratio_db, i)
        imwrite(save_path, im)
