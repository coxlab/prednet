from PIL import Image
import sys

def processImage(infile, save_dir='./'):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGB", im.size)
            new_im.paste(im)
            new_im.save(save_dir + '/foo'+str(i)+'.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

if __name__ == '__main__':
    processImage('./figs/flash_lag.gif', save_dir='./figs/')
