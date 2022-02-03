# convert degree color to the input of RNN and also the output of RNN to the degree color. Basically this file contains color encoder and decoder.
# Currently we have:
# 1. Encode/Decode degree to LMS space
# 2. Encode/Decode degree to the firing rate of color cell (virtual) (under construction).
import numpy as np
import colormath.color_objects as colorob
import colormath.color_conversions as colorcv
from scipy.stats import vonmises
from matplotlib import cm

# normalized to D65
MAT_LMS_NORM = np.array([[0.4002, -0.2263, 0], [0.7076, 1.1653, 0], [-0.0808, 0, 0.9182]])
MAT_LMS_NORM_INV = np.array(
[[ 1.86006661,  0.36122292,  0.        ],
 [-1.12948008,  0.63880431, -0.        ],
 [ 0.16368262,  0.03178699,  1.08908734]]
)

class Degree_color():
    '''
    output several format of color which is defined by a circle in cielab space.
    '''
    def __init__(self, center_l=80, center_a=22, center_b=14, radius=60):
        '''
        center_l, center_a, center_b (int): the center of color circle
        radius (float): the raius of color circle
        The default value comes from experiment 1a from paper error-correcting ...
        '''
        self.center_l, self.center_a, self.center_b = center_l, center_a, center_b
        self.radius = radius
    def out_color(self, degree, fmat='LAB', is_upscaled=False):
        '''
        input:
          degree (np.array or int): the unit is degree. if the input is int, the output color will have degrees np.arange(0, 360, degree)
          fmat (str): output color format, can be LAB, RGBA (where A is fixed as 1), LMS
          is_upscaled (bool): only for RGB color. It true, the range wouls be 0-255, false then 0-1
        output:
          color ([n, c] matrix): where n is the length of degree, c is the number of channel in of the color format.
        '''
        # calculate the color on LAB space
        i = 0
        if not hasattr(degree, '__iter__'):
            degree = np.linspace(0, 360, degree)

        lab = self.deg2lab(degree)
        if fmat == 'LAB':
            return lab
        elif fmat == 'RGBA':
            return lab2rgba(lab)
        elif fmat == 'RGB':
            return lab2rgb(lab, is_upscaled=is_upscaled)
        elif fmat == 'XYZ':
            return lab2xyz(lab)
        elif fmat == 'LMS':
            xyz = lab2xyz(lab)
            return xyz2lms(xyz)
        elif fmat == 'RGBA_my':
            cmap = cm.get_cmap('hsv')
            degree_norm = degree / 360.0
            color_list = cmap(degree_norm)
            #alpha = 0.3
            #color_list[:, -1] = alpha * np.ones(color_list.shape[0])
            #light = 0.6
            #color_list[:, 0:3] *= light
            return color_list

    def set_centers(self, center_l, center_a, center_b):
        self.center_l, self.center_a, self.center_b = center_l, center_a, center_b
    def set_radius(self, radius):
        self.radius = radius

    def deg2lab(self, degree):
        rads = np.deg2rad(degree) # convert to rad so that can be calculated by np.cos

        n_sample = len(rads)
        lab = np.zeros((n_sample, 3)) # 3 channels l, a, b
        i = 0
        for rad in rads:
            lab[i, :] = np.array([self.center_l, self.center_a + self.radius * np.cos(rad), self.center_b + self.radius * np.sin(rad)])
            i = i + 1
        return lab
    def lab2deg(self, lab):
        n_sample = lab.shape[0]
        degree = np.zeros(n_sample)
        for i in range(n_sample):
            l, a, b = lab[i]
            temp_sin = (b - self.center_b) / self.radius
            temp_cos = (a - self.center_a) / self.radius
            loc = np.arctan2(temp_sin, temp_cos)
            loc = np.mod(loc, 2*np.pi)
            degree[i] = loc
        degree = degree / (2 * np.pi) * 360
        return degree
    def deg2rgba(self, degree):
        lab = self.deg2lab(degree)
        return lab2rgba
    def lms2deg(self, lms):
        '''
        lms (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, m, s
        '''
        xyz = lms2xyz(lms)
        lab = xyz2lab(xyz)
        degree = self.lab2deg(lab)
        return degree

def lab2rgba(lab):
    '''
    lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    '''
    n_sample = lab.shape[0]
    lab_instance = colorob.LabColor(0, 0, 0, observer='2', illuminant='d65')
    RGBA = np.zeros((n_sample, 4)) # 4 channels R, G, B, A
    for i in range(n_sample):
        lab_instance.lab_l, lab_instance.lab_a, lab_instance.lab_b = lab[i, :]
        sRGB_sample = colorcv.convert_color(lab_instance, colorob.sRGBColor)
        RGBA[i, :3] = sRGB_sample.clamped_rgb_r, sRGB_sample.clamped_rgb_g, sRGB_sample.clamped_rgb_b
        RGBA[i, 3] = 1
    return RGBA

def lab2rgb(lab, is_upscaled=False):
    '''
    lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    '''
    n_sample = lab.shape[0]
    lab_instance = colorob.LabColor(0, 0, 0, observer='2', illuminant='d65')
    RGB = np.zeros((n_sample, 3)) # 4 channels R, G, B, A
    for i in range(n_sample):
        lab_instance.lab_l, lab_instance.lab_a, lab_instance.lab_b = lab[i, :]
        sRGB_sample = colorcv.convert_color(lab_instance, colorob.sRGBColor)
        if is_upscaled:
            sRGB_sample = sRGB_sample.get_upscaled_value_tuple()
            RGB[i] = [ min(rgb, 255) for rgb in sRGB_sample ] # clamp to 0 to 255
        else:
            RGB[i] = sRGB_sample.clamped_rgb_r, sRGB_sample.clamped_rgb_g, sRGB_sample.clamped_rgb_b
    return RGB

def lab2xyz(lab):
    '''
    input:
      lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    return:
      xyz (n * 3 matrix): where n is the number of colors, 3 means the three channels of x, y, z
    '''
    n_sample = lab.shape[0]
    lab_instance = colorob.LabColor(0, 0, 0, observer='2', illuminant='d65')
    xyz = np.zeros((n_sample, 3)) # 4 channels R, G, B, A
    for i in range(n_sample):
        lab_instance.lab_l, lab_instance.lab_a, lab_instance.lab_b = lab[i, :]
        xyz_sample = colorcv.convert_color(lab_instance, colorob.XYZColor)
        xyz[i, :] = xyz_sample.get_value_tuple()
    return xyz

def xyz2lms(xyz):
    '''
    xyz (n * 3 matrix): where n is the number of colors, 3 means the three channels of x, y, z
    '''
    lms = np.dot(MAT_LMS_NORM, xyz.T)
    # reshape to 2-D image
    lms = lms.T.reshape(xyz.shape)
    return lms

def lms2xyz(lms):
    '''
    lms (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, m, s
    '''
    xyz = np.dot(MAT_LMS_NORM_INV, lms.T)
    xyz = xyz.T.reshape(lms.shape)
    return xyz

def xyz2lab(xyz):
    '''
    input:
      xyz (n * 3 matrix): where n is the number of colors, 3 means the three channels of x, y, z
    return:
      lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    '''
    xyz = xyz.reshape((-1, 3))
    n_sample = xyz.shape[0]
    xyz_instance = colorob.XYZColor(0, 0, 0, observer='2', illuminant='d65')
    lab = np.zeros((n_sample, 3))
    for i in range(n_sample):
        xyz_instance.xyz_x, xyz_instance.xyz_y, xyz_instance.xyz_z = xyz[i, :]
        lab_sample = colorcv.convert_color(xyz_instance, colorob.LabColor)
        lab[i, :] = lab_sample.get_value_tuple()
    return lab
