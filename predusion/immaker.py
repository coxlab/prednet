'''
Create a series of static images to the target folder
'''
import numpy as np
import matplotlib.pyplot as plt
from imageio import imsave


class Immaker():
    def __init__(self, imshape=(128, 160), background=255):
        self.imshape = imshape
        self.im = []
        self.background = background

    def save_fig(self, save_path='./out.png'):
        imsave(save_path, self.im)

class Grid(Immaker):

    def set_grid(self, offset=[0, 0], frac_d=5, ratio_db=1.5, rgb_color=[0, 0, 0]):
        '''
        offset: the left upper coner of nonzero pixel.
        frac_d: the length of square is equal to imashape[0] / frac_d
        ratio_db: The spacing of two adjacent squares is b = int(ratio_db * d) which must be larger than d
        '''
        assert ratio_db >=1, 'spacing of squares in the grid, ratio_db must larger than 1'

        self.offset = np.uint8(offset)
        self.frac_d = frac_d
        self.ratio_db = ratio_db

        im = np.ones((*self.imshape, 3), dtype=np.uint8) * 255
        d = self.imshape[0] // frac_d # width, measured in the number of pixels in each square
        b = int(ratio_db * d) # spacing of two adjacent squares, this must be larger than d

        for idy in range(offset[0], self.imshape[0], b):
            for idx in range(offset[1], self.imshape[1], b):
                try:
                    im[idy:idy+d, idx:idx+d] = rgb_color
                except:
                    idy_end = min(idy+d, self.imshape[0])
                    idx_end = min(idx+d, self.imshape[1])
                    im[idy:idy_end, idx:idx_end] = rgb_color
        self.im = im

class Square(Grid):

    def set_square(self, center=None, width=None, rgb_color=(0, 0, 0)):
        '''
        center [arr like, 2, numerical]: the center of the square. Measured using pixel, with the left upper coner as (0, 0). None means center of the image
        width [numerical]: width of the square measured in pixel
        '''

        if center is None: center = np.array(self.imshape, dtype=int) // 2
        if width is None: width = self.imshape[0] // 2

        x0, y0 = center - width // 2

        assert (x0 > 0) & (y0 > 0), 'The square is too large'

        im = np.ones((*self.imshape, 3), dtype=np.uint8) * self.background


        try:
            im[x0:x0+width, y0:y0+width] = rgb_color
        except:
            x_end = min(x0+width, self.imshape[0])
            y_end = min(y0+width, self.imshape[1])
            im[x0:x_end, y0:y_end] = rgb_color

        self.im = im


if __name__ == '__main__':
    n_image = 5

    frac_d = 2
    ratio_db = 3
    imshape = (128, 160)

    grid = Grid(imshape)
    grid.set_grid(offset=[64, 80], frac_d=frac_d, ratio_db=ratio_db)

    for i in range(n_image):
        save_path = './figs/grid_{0}_{1}_{2}.png'.format(frac_d, ratio_db, i)
        grid.save_fig(save_path=save_path)

    center = 10
    width = 10
    square = Square(imshape)
    square.set_square()

    for i in range(n_image):
        save_path = './figs/square_{0}_{1}_{2}.png'.format(center, width, i)
        square.save_fig(save_path=save_path)
