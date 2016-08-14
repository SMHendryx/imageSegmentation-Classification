__author__ = 'seanhendryx'

# Sean Hendryx
# This script segments an image into the input number of segments using the Simple Linear Iterative Clustering (SLIC) algorithm, which is an adaptation of k-means to efficiently generate superpixels 
# Script built to run on Python version 3.4
# References: Segmentation: A SLIC Superpixel Tutorial using Python <http://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/>

import numpy
import skimage
from skimage import io
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def main():

    filename = 'mesquitesFloat.png'
    img = io.imread(filename)

    #CONVERSIONS:
    #convert to lab color space
    #img = skimage.color.rgb2lab(img)
    #img = img_as_float(img)
    #io.imsave('mesquitesFloat.png', img)

    #print(img.shape)

    plt.figure(1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')


    # loop over the number of segments
    for numSegments in (100, 200, 300):
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(img, n_segments = numSegments, sigma = 1)

        # show the output of SLIC
        fig = plt.figure("Superpixels of -- %d segments" % (numSegments))
        subplot = fig.add_subplot(1, 1, 1)
        subplot.imshow(mark_boundaries(img, segments))
        plt.axis("off")



    plt.show()
    #plt.savefig("test.png",bbox_inches='tight')



# Main Function
if __name__ == '__main__':
    main()