import os
import glob
import PIL
from PIL import Image

basewidth = 300
indir = 'c:\\dev\\code\\MachineLearning\\Image\\*.jpg'
for filename in glob.glob(indir): # loop through each file
    img = Image.open(filename)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    img.save((filename.replace(".JPG","")) + "_shrink.jpg")
