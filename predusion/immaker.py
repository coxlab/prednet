'''
Create a series of static images to the target folder
'''
import numpy as np
import matplotlib.pyplot as plt
from imageio import imsave
from scipy.ndimage import gaussian_filter


class Immaker():
    def __init__(self, imshape=(128, 160), background=255):
        self.imshape = imshape
        self.im = []
        self.background = background

    def save_fig(self, save_path='./out.png'):
        imsave(save_path, self.im)

    def get_im(self):
        return self.im.copy()

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

    def set_full_square(self, color=[0, 0, 0]):
        '''
        obtain a full colored square with self.imshape
        color (numeric array [3]): the value of color should within 0 to 255
        '''
        self.im = np.ones((*self.imshape, 3), dtype=np.uint8)

        self.im[:, :, 0] = np.uint8(color[0]) # add the color
        self.im[:, :, 1] = np.uint8(color[1])
        self.im[:, :, 2] = np.uint8(color[2])

        return self.im.copy()

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

        return self.im.copy()

class Seq_gen():
    '''
    generate image sequence as input to the prednet
    '''
    @staticmethod
    def repeat(im, n):
        '''
        repeat images for n times
        im (numpy array, numeric, [width, height, 3]): the final 3 channels are the RGB values of the image
        '''
        return np.repeat(im[None, :, :], int(n), axis=0)

class Batch_gen():

    @staticmethod
    def color_noise_full(imshape, n_image=1, batch_size=1, is_upscaled=True, sig=None):
        '''
        generate color white noise square, each pixel follows the uniform distribution
        input:
          n_image (int): the number of images
          imshape (array, int, [width, height])
        output:
          w_square (array, float, [batch_size, n_image, imshape[0], imshape[1], 3]): is upscaled means the value of each pixel ranges from 0 to 255, otherwise 0 to 1
        '''
        w_image = np.random.uniform(size=(batch_size, n_image, *imshape, 3))
        w_image = Batch_gen()._gaussian_filter(w_image, sig)
        if is_upscaled:
            return np.uint8(w_image * 255)
        else:
            return w_image

    @staticmethod
    def grey_noise_full(imshape, n_image=1, batch_size=1, is_upscaled=True, sig=None):
        '''
        generate color white noise square, each pixel follows the uniform distribution
        input:
          n_image (int): the number of images
          imshape (array, int, [width, height])
          batch_size (int)
          sig (array like, float, [3]): see decription in low pass filter to the images _gaussian_filter(). If None means do not filt
        output:
          w_square (array, float, [n_image, imshape[0], imshape[1], 3]): is upscaled means the value of each pixel ranges from 0 to 255, otherwise 0 to 1
        '''
        w_image = np.random.uniform(size=(batch_size, n_image, *imshape, 1))
        w_image = np.repeat(w_image, 3, axis=4)
        w_image = Batch_gen()._gaussian_filter(w_image, sig)

        if is_upscaled:
            return np.uint8(w_image * 255)
        else:
            return w_image

    @staticmethod
    def _gaussian_filter(img, sig=None):
        '''
        spatial temporal filter of a batch
        images (numpy array, float, same shape as the output of grey_noise_full):
        sig (array like, float, [3]): the sd of the gaussian filter. The first correponds to the temporal filter, final two elements correspond to the spatial part. If None means do not filt
        '''
        if sig is None:
            return img
        else:
            img_filt = gaussian_filter(img, sigma=(0, *sig, 0))
            return img_filt

if __name__ == '__main__':
    import matplotlib.pyplot as plt

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

    # generate color noise images
    w_img = Batch_gen().color_noise_full(imshape, 3, 2)

    plt.figure()
    for im in w_img[1]:
        plt.imshow(im)
        plt.show()

    # generate grey noise images
    w_img = Batch_gen().grey_noise_full(imshape, 3, 2)

    plt.figure()
    for im in w_img[1]:
        plt.imshow(im)
        plt.show()

    imshape = (128, 160)
    # generate grey noise images
    w_img = Batch_gen().grey_noise_full(imshape, 3, 2, sig=(0, 2, 2))

    plt.figure()
    for im in w_img[1]:
        plt.imshow(im)
        plt.show()
