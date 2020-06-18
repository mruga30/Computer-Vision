
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image

h = 0
w = 0

def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')
    else:
	#size by size ndarray of type float32 is created
    	k = np.empty([size, size], dtype = 'float32')
    	ctr = (size/2)+0.5
    	idx = (size/2)-0.5
    	u = -1*idx
    	v = idx

    	#the values of the kernel are calculated
    	for x in range(0, size):
    		for y in range(0, size):
    			k[x,y] = np.float32((1/2*np.pi*(sigma**2))*(np.exp(-(((u**2)+(v**2))/(sigma**2)))))
    			u = u + 1
    		v = v - 1
    		u = -idx
 	#Values are normalised such that the sum of the kernel = 1
    	sum = np.sum(k)
    	rv = k/sum
    	return rv


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))

    #Following are calculate to get: the ith and jth locations to start and stop at
    i_kernel = ((kernel.shape[0])/2)+0.5-1
    j_kernel = ((kernel.shape[1])/2)+0.5-1
    i_kernel_bnd = img.shape[0]-i_kernel
    j_kernel_bnd = img.shape[1]-j_kernel

    #The start and stop bounds are compared to input pixels to check if the kernel stretches beyond the border of the image.
    if ((i < i_kernel)or(j<j_kernel)or(i>(i_kernel_bnd-1))or(j>(j_kernel_bnd-1))):
        #if true, the pixel value is returned as it is
    	resultPix = img[i,j]
    	return resultPix
    else:
        #else, perform convolution
    	#first we calculate the upper and lower bounds where convolution is to be performed according to the pixel value
    	low_i = np.int16(i-i_kernel)
    	up_i = np.int16(i+i_kernel)
    	low_j = np.int16(j-j_kernel)
    	up_j = np.int16(j+j_kernel)
    	k_i = 0
    	k_j = 0
    	resultPix = 0
    	#use the bounds to perform convolution
    	for m in range(low_i,up_i+1):
    		for n in range(low_j,up_j+1):
    			resultPix = resultPix + img[m,n]*kernel[k_i,k_j]
    			k_j = k_j+1
    		k_i = k_i+1
    		k_j = 0
    	resultPix = round(resultPix,4)
    	return resultPix

def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    #Made a copy of the input image to save results
    imgCopy = np.copy(img)
    #Populated each pixel in the input image by calling convolve_pixel
    resultConv = np.empty([img.shape[0], img.shape[1]], dtype = 'uint8')
    for x in range(0,img.shape[0]):
    	for y in range(0,img.shape[1]):
    		resultConv[x,y]=convolve_pixel(img, kernel, x, y) 
    return resultConv

def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    else:
    	#image is split into three channels-r,g,b by splitting the 3D image array into 2D arrays
        b = img[:,:,2]
        g = img[:,:,1]
        r = img[:,:,0]
    return r,g,b


def merge(r, g, b):
    """
    # Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    # Args:
        # r:    A height x width ndarray of red pixel values.
        # g:    A height x width ndarray of green pixel values.
        # b:    A height x width ndarray of blue pixel values.

    # Returns:
        # A height x width x 3 ndarray representing the color image.
    # """
    #the r,g,b channels are merged by converting 3 2d arrays into one 3d array 
    result = np.empty([h,w,3], dtype = 'uint8')
    result = np.dstack((r,g,b))
    return result


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--sigma', type=float, default=1.0,help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,help='The size of the kernel.')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)
    #print(inputImage.shape)
    h = inputImage.shape[0]
    w = inputImage.shape[1]

    # Split it into three channels
    logging.info('Splitting it into 3 channels')
    (r, g, b) = split(inputImage)
    #imageio.imwrite(r, resultImage)


    # compute the gaussian kernel
    logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                 (args.k, args.sigma))
    kernel = create_gaussian_kernel(args.k, args.sigma)
    #print(kernel)

    # convolve it with each input channel
    logging.info('Convolving the first channel')
    r = convolve(r, kernel)
    logging.info('Convolving the second channel')
    g = convolve(g, kernel)
    logging.info('Convolving the third channel')
    b = convolve(b, kernel)

    # merge the channels back
    logging.info('Merging results')
    resultImage = merge(r, g, b)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
