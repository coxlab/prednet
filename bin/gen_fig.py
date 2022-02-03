# generate figs for processing
import predusion.immaker as immaker
from predusion.color_deg import Degree_color
import numpy as np
import os

n_image, n_image_w = 4, 4

imshape = (128, 160)

l, a, b, r = 80, 22, 14, 70
n_deg = 30
deg_color = Degree_color(center_l=l, center_a=a, center_b=b, radius=r)

center = 10
width = int(imshape[0] / 2)

degree = np.linspace(0, 360, n_deg)
color_list = deg_color.out_color(degree, fmat='RGB', is_upscaled=True)
square = immaker.Square(imshape, background=150)

#color_list = np.zeros((30, 3))
#i = 0
#for ci in np.arange(0, 1.25, 0.25):
#    color_list[i] = [ci, 1, 0]
#    color_list[i + 1] = [0, 1, ci]
#    color_list[i + 2] = [0, ci, 1]
#    color_list[i + 3] = [ci, 0, 1]
#    color_list[i + 4] = [1, 0, ci]
#    color_list[i + 5] = [1, ci, 0]
#    i = i + 6
#color_list = color_list * 255

for color in color_list:
    r, g, b = color
    square.set_square(width=width, rgb_color=color)
    save_path_dir = './kitti_data/raw/square_{center}_{width}/square_{r}_{g}_{b}'.format(center=center, width=width, r=str(round(r, 2)), g=str(round(g, 2)), b=str(round(b, 2)))
    if not os.path.exists(save_path_dir): os.makedirs(save_path_dir)

    # generate n_image colored square
    for i in range(n_image):
        save_path = os.path.join(save_path_dir, 'square{i}.png'.format(i=str(i).zfill(2)))
        square.save_fig(save_path=save_path)

    # generate n_image_w white squares
    bg = square.background
    square.set_square(width=width, rgb_color=(bg, bg, bg))
    for i in range(n_image, n_image + n_image_w):
        save_path = os.path.join(save_path_dir, 'square{i}.png'.format(i=str(i).zfill(2)))
        square.save_fig(save_path=save_path)
