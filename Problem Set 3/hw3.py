
import pickle

import numpy as np
import argparse
import imageio
import logging
import sys
from scipy.ndimage.filters import convolve

# Displacements are by default saved to a file after every run. Once you have confirmed your
# LK code is working, you can load saved displacements to save time testing the
# rest of the project.
DEFAULT_DISPLACEMENTS_FILE = "final_displacements.pkl"

def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0]-1, image.shape[1]-1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0]-2, image.shape[1]-2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl+1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1-a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1-a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1-b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize//2), ksize//2, ksize)
                    ** 2 / 2) / np.sqrt(2*np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""

    # Cylindrical warping introduces black pixels which should be ignored, and
    # motion in dark regions is difficult to estimate. Generate a binary mask
    # indicating pixels that are valid (average color value > 0.25) in both H
    # and I.
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]

    # Compute the partial image derivatives w.r.t. X, Y, and Time (t).
    # In other words, compute I_y, I_x, and I_t
    # To achieve this, use a _normalized_ 3x3 sobel kernel and the convolve_img
    # function above. NOTE: since you're convolving the kernel, you need to 
    # multiply it by -1 to get the proper direction.
    kernel_x = np.array([[-0.125,0,0.125],[-0.25,0,0.25],[-0.125,0,0.125]])
    kernel_y = np.array([[0.125,0.25,0.125],[0,0,0],[-0.125,-0.25,-0.125]])
    I_x = -convolve_img(I, kernel_x)
    I_y = convolve_img(I, kernel_y)
    I_t = I-H

    # Compute the various products (Ixx, Ixy, Iyy, Ixt, Iyt) necessary to form
    # AtA. Apply the mask to each product.
    Ixx = I_x*I_x
    Ixy = I_x*I_y
    Iyy = I_y*I_y
    Ixt = I_x*I_t
    Iyt = I_y*I_t

    Ixx = Ixx*mask
    Ixy = Ixy*mask
    Iyy = Iyy*mask
    Ixt = Ixt*mask
    Iyt = Iyt*mask

    # Build the AtA matrix and Atb vector. You can use the .sum() function on numpy arrays to help.
    AtA = np.array([[np.sum(Ixx),np.sum(Ixy)],[np.sum(Ixy),np.sum(Iyy)]])
    Atb = np.array([-np.sum(Ixt),-np.sum(Iyt)])

    # Solve for the displacement using linalg.solve
    displacement = np.linalg.solve(AtA,Atb)

    # return the displacement and some intermediate data for unit testing..
    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float32)
    #disp1 = disp.flatten()
    for i in range(steps):
        # Translate the H image by the current displacement (using the translate function above)
    	H1 = translate(H, disp)
        # run Lucas Kanade and update the displacement estimate
    	lk_return = lucas_kanade(H1,I)
    	disp = disp+lk_return[0]
    # Return the final displacement
    return disp

def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Retuns:
        An array of images where each image is a blurred and shruken version of the first.
    """

    # Compute a gaussian kernel using the gaussian_kernel function above. You can leave the size as default.
    kernel = gaussian_kernel(5)

    # Add image to the the list as the first level
    pyr = [image]
    NI = image
    for level in range(1, levels):
        # Convolve the previous image with the gussian kernel
    	nI = convolve_img(NI,kernel)

        # decimate the convolved image by downsampling the pixels in both dimensions.
        # Note: you can use numpy advanced indexing for this (i.e., ::2)
    	NI = nI[::2,::2]

        # add the sampled image to the list
    	pyr.append(NI)
    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    initial_d = np.asarray(initial_d, dtype=np.float32)

    # Build Gaussian pyramids for the two images.
    Hg = gaussian_pyramid(H, levels)
    Ig = gaussian_pyramid(I, levels)

    # Start with an initial displacement (scaled to the coarsest level of the
    # pyramid) and compute the updated displacement at each level using Lucas
    # Kanade.
    disp = initial_d / 2.**(levels)
    for level in range(levels):
        # Get the two images for this pyramid level.
    	Hp = Hg[-level-1]
    	Ip = Ig[-level-1]

        # Scale the previous level's displacement and apply it to one of the
        # images via translation.
    	disp = disp*2
    	Ip = translate(Ip, -disp)

        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
    	displacement = iterative_lucas_kanade(Hp,Ip,steps)

        # Update the displacement based on the one you just computed.
    	disp += displacement

    # Return the final displacement.
    return disp


def build_panorama(images, shape, displacements, initial_position, blend_width=16):
    # Allocate an empty floating-point image with space to store the panorama
    # with the given shape.
    image_height, image_width = images[0].shape[:2]
    pano_height, pano_width = shape
    panorama = np.zeros((pano_height, pano_width, 3), np.float32)

    # Place the last image, warped to align with the first, at its proper place
    # to initialize the panorama.
    cur_pos = initial_position
    cp = np.round(cur_pos).astype(np.int32)
    panorama[cp[0]: cp[0] + image_height, cp[1]: cp[1] +
             image_width] = translate(images[-1], displacements[-1])

    # Place the images at their final positions inside the panorama, blending
    # each image with the panorama in progress. Use a blending window with the
    # given width.
    for i in range(len(images)):
        cp = np.round(cur_pos).astype(np.int32)

        overlap = image_width - abs(displacements[i][0])
        blend_start = int(overlap / 2 - blend_width / 2)
        blend_start_pano = int(cp[1] + blend_start)

        pano_region = panorama[cp[0]: cp[0] + image_height,
                               blend_start_pano: blend_start_pano+blend_width]
        new_region = images[i][:, blend_start: blend_start+blend_width]

        mask = np.zeros((image_height, blend_width, 1), np.float32)
        mask[:] = np.linspace(0, 1, blend_width)[np.newaxis, :, np.newaxis]
        mask[np.all(new_region == 0, axis=2)] = 0
        mask[np.all(pano_region == 0, axis=2)] = 1

        blended_region = mask * new_region + (1-mask) * pano_region

        blended = images[i].copy("C")
        blended[:, blend_start: blend_start+blend_width] = blended_region
        blended[:, :blend_start] = panorama[cp[0] : cp[0] + image_height, cp[1]: blend_start_pano]

        panorama[cp[0]: cp[0] + blended.shape[0],
                 cp[1]: cp[1] + blended.shape[1]] = blended
        cur_pos += -displacements[i][::-1]
        print("Placed %d." % i)

    # Return the finished panorama.
    return panorama

def mosaic(images, initial_displacements, load_displacements_from):
    """Given a list of N images taken in clockwise order and corresponding
    initial X/Y displacements of shape (N,2), refine the displacements and
    build a mosaic.

    initial_displacement[i] gives the translation that should be appiled to
    images[i] to align it with images[(i+1) % N]."""
    N = len(images)

    if load_displacements_from:
        print("Loading saved displacements...")
        final_displacements = pickle.load(open(load_displacements_from, "rb"))
    else:
        print("Refining displacements with Pyramid Iterative Lucas Kanade...")
        final_displacements = []
        for i in range(N):
            # TODO Use Pyramid Iterative Lucas Kanade to compute displacements from
            # each image to the image that follows it, wrapping back around at
            # the end. A suggested number of levels and steps is 4 and 5
            # respectively. Make sure to append the displacement to
            # final_displacements so it gets saved to disk if desired.
    	    fdisp = pyramid_lucas_kanade(images[i],images[(i+1)%N],initial_displacements[i],4,5)
    	    final_displacements.append(fdisp)
    	    # Some debugging output to help diagnose errors.
    	    print("Image %d:" % i,
                  initial_displacements[i], "->", final_displacements[i], "  ",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i+1) % N], -initial_displacements[i]))).mean(), "->",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i+1) % N], -final_displacements[i]))).mean()
                  )
        print('Saving displacements to ' + DEFAULT_DISPLACEMENTS_FILE)
        pickle.dump(final_displacements, open(DEFAULT_DISPLACEMENTS_FILE, "wb"))


    # Use the final displacements and the images' shape compute the full
    # panorama shape and the starting position for the first panorama image.

    pano_width = 0
    height_disp = 0
    for i in range(N):
    	pano_width += images[i].shape[1]+final_displacements[i][0]
    	height_disp +=final_displacements[i][1]

    pano_height = int(images[0].shape[0]+height_disp) #+47.5724956)
    pano_width = int(pano_width)

    initial_pos = np.array([height_disp,0.0])

    # Build the panorama.
    print("Building panorama...")
    panorama = build_panorama(images, (pano_height, pano_width), final_displacements, initial_pos.copy())
    return panorama, final_displacements


def warp_panorama(images, panorama, final_displacements):
    # Extra credit: Implement this function!

    # Resample the panorama image using a linear warp to distribute any vertical
    # drift to all of the sampling points. The final height of the panorama should
    # be equal to the height of one of the images.

    # Crop the panorama horizontally so that the left and right edges of the
    # panorama match (making it form a loop).

    # Return your final corrected panorama.
    warped = panorama
    return warped


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Creates a mosaic by stitching together a provided set of images.')
    parser.add_argument(
        'input', type=str, help='A txt file containing the images and initial displacement positions.')
    parser.add_argument('output', type=str,
                        help='What image file to save the panorama to.')
    parser.add_argument('--displacements', type=str,
                        help='Load displacements from this pickle file (useful for build_panorama).', default=None)
    args = parser.parse_args()

    filenames, xinit, yinit = zip(
        *[l.strip().split() for l in open(args.input).readlines()])
    xinit = np.array([float(x) for x in xinit])[:, np.newaxis]
    yinit = np.array([float(y) for y in yinit])[:, np.newaxis]
    disps = np.hstack([xinit, yinit])

    images = [imageio.imread(fn)[:, :, :3].astype(
        np.float32)/255. for fn in filenames]

    panorama, final_displacements = mosaic(images, disps, args.displacements)

    result = warp_panorama(images, panorama, final_displacements)
    imageio.imwrite(args.output, result)
