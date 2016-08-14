__author__ = 'seanhendryx'

# Sean Hendryx

# Script built to run on Python version 3.4
# References: Sources on transforming data into a workable format:
# https://www.quora.com/How-can-my-pixel-data-from-an-image-be-outputted-into-a-CSV-file-like-this-in-Python
# https://sourcedexter.com/2013/08/27/extracting-pixel-values-of-an-image-in-python/

import numpy
import skimage
from skimage import io
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image

def main():
    """
    This script loads an image, extracts the RGB values from each pixel, regardless of spatial information, and saves the values to a csv (by first converting to a numpy array)
    :return:
    """

    #filename = 'mesquites.png'
    im = Image.open('mesquitesSubsetNoAlpha.png')

    #TRANSFORM AND SAVE PIXELS VALUES AS CSV
    pixVals = list(im.getdata())
    #flatPixVals = [x for sets in pixVals for x in sets]
    pixels = numpy.asarray(pixVals)
    numpy.savetxt("mesquitesSubsetPixelData.csv", pixels, delimiter = ",")





# Main Function
if __name__ == '__main__':
    main()