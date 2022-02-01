# generate figs for processing
import predusion.immaker as immaker
from predusion.color_deg import Degree_color
import os

n_image, n_image_w = 4, 4

imshape = (128, 160)

l, a, b, r = 80, 22, 14, 52
deg_color = Degree_color(center_l=l, center_a=a, center_b=b, radius=r)
degree = [100]
color_list = deg_color.out_color(degree, fmat='RGB', is_upscaled=True)

center = 10
width = int(imshape[0] / 1.8)
square = immaker.Square(imshape)

for color in color_list:
    r, g, b = color
    square.set_square(width=width, rgb_color=color)
    save_path_dir = './kitti_data/raw/square_{center}_{width}/square_{r}_{g}_{b}'.format(center=center, width=width, r=str(round(r, 2)), g=str(round(g, 2)), b=str(round(b, 2)))
    if not os.path.exists(save_path_dir): os.makedirs(save_path_dir)

    # generate n_image colored square
    for i in range(n_image):
        save_path = os.path.join(save_path_dir, 'square{i}.png'.format(i=i))
        square.save_fig(save_path=save_path)

    # generate n_image_w white squares
    square.set_square(width=width, rgb_color=(255, 255, 255))
    for i in range(n_image, n_image + n_image_w):
        save_path = os.path.join(save_path_dir, 'square{i}.png'.format(i=i))
        square.save_fig(save_path=save_path)
