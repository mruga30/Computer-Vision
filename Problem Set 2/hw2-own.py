
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def build_A(pts1, pts2):
    """
    Constructs the intermediate matrix A used in the total least squares 
    computation of an homography mapping pts1 to pts2.

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 2Nx9 matrix A that we'll use to solve for h
    """
    if pts1.shape != pts2.shape:
        raise ValueError('The source points for homography computation must have the same shape (%s vs %s)' % (
            str(pts1.shape), str(pts2.shape)))
    if pts1.shape[0] < 4:
        raise ValueError('There must be at least 4 pairs of correspondences.')
    num_pts = pts1.shape[0]

    # TODO: Create A which is 2N by 9...
    A = np.empty([8, 9], dtype = 'float32')
    # TODO: iterate over the points and populate the rows of A.
    p = 0
    for i in [0,2,4,6]:
    	A[i,0]=pts1[p,0]
    	A[i,1]=pts1[p,1]
    	A[i,2]=1
    	A[i,3]=0
    	A[i,4]=0
    	A[i,5]=0
    	A[i,6]=-pts1[p,0]*pts2[p,0]
    	A[i,7]=-pts1[p,1]*pts2[p,0]
    	A[i,8]=-pts2[p,0]
    	i = i+1
    	A[i,0]=0
    	A[i,1]=0
    	A[i,2]=0
    	A[i,3]=pts1[p,0]
    	A[i,4]=pts1[p,1]
    	A[i,5]=1
    	A[i,6]=-pts1[p,0]*pts2[p,1]
    	A[i,7]=-pts1[p,1]*pts2[p,1]
    	A[i,8]=-pts2[p,1]
    	p=p+1
    return A


def compute_H(pts1, pts2):
    """
    Computes an homography mapping one set of co-planar points (pts1)
    to another (pts2).

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 3x3 homography matrix that maps homogeneous coordinates of pts 1 to those in pts2.
    """
    # TODO: Construct the intermediate A matrix using build_A
    A = build_A(pts1,pts2)

    # TODO: Compute the symmetric matrix AtA.
    At = A.transpose()
    AtA = np.dot(At, A)

    # TODO: Compute the eigenvalues and eigenvectors of AtA.
    eig_vals, eig_vecs =  np.linalg.eig(AtA)

    # TODO: Determine which eigenvalue is the smallest
    idx = np.argsort(eig_vals)
    min_eig_val_index = eig_vals[idx]

    # TODO: Return the eigenvector corresponding to the smallest eigenvalue, reshaped as a 3x3 matrix.
    sort_eig_vec = eig_vecs[:,idx]
    h = sort_eig_vec[:,0]
    min_eig_vec = h.reshape(3,3)

    return min_eig_vec


def bilinear_interp(image, point):
    """
    Looks up the pixel values in an image at a given point using bilinear
    interpolation. point is in the format (x, y).

    Args:
        image:      The image to sample
        point:      A tuple of floating point (x, y) values.

    Returns:
        A 3-dimensional numpy array representing the pixel value interpolated by "point".
    """
    # TODO: extract x and y from point
    x = point[0]
    y = point[1]

    # TODO: Compute i,j as the integer parts of x, y
    round = np.array([x,y])
    round1 = np.floor(round)
    round2= round1.astype(int)
    i = round2[0]
    j = round2[1]

    # TODO: check that i + 1 and j + 1 are within range of the image. if not, just return the pixel at i, j
    if (((i+1)>(image.shape[0]-1)) or ((j+1)>(image.shape[1]-1))):
                a = 0
                b = 0
                value = ((1-a)*(1-b)*image[j,i])+(a*(1-b)*image[j,i])+(a*b*image[j,i])+((1-a)*b*image[j,i])
    else:
    	# TODO: Compute a and b as the floating point parts of x, y
    	a = x-i
    	b = y-j

    	# TODO: Take a linear combination of the four points weighted according to the inverse area around them (i.e., the formula for bilinear interpolation)
    	value = ((1-a)*(1-b)*image[j,i])+(a*(1-b)*image[j,i+1])+(a*b*image[j+1,i+1])+((1-a)*b*image[j+1,i])
    return value


def apply_homography(H, points):
    """
    Applies the homography matrix H to the provided cartesian points and returns the results 
    as cartesian coordinates.

    Args:
        H:      A 3x3 floating point homography matrix.
        points: An Nx2 matrix of x,y points to apply the homography to.

    Returns:
        An Nx2 matrix of points that are the result of applying H to points.
    """
    n = points.shape[0]
    # TODO: First, transform the points to homogeneous coordinates by adding a `1`
    ones = np.empty([n, 1], dtype = 'float32')
    for i in range(n):
    	ones[i,0] = 1
    	points1=np.append(points, ones, axis=1)

    # TODO: Apply the homography
    result = np.empty([n, 3], dtype = 'float32')
    for i in range(n):
    	result[i,2]= H[2,0]*points1[i,0]+H[2,1]*points1[i,1]+H[2,2]*points1[i,2]
    	result[i,0]= (H[0,0]*points1[i,0]+H[0,1]*points1[i,1]+H[0,2]*points1[i,2])/result[i,2]
    	result[i,1]= (H[1,0]*points1[i,0]+H[1,1]*points1[i,1]+H[1,2]*points1[i,2])/result[i,2]

    # TODO: Convert the result back to cartesian coordinates and return the results
    result1 = np.delete(result,2,axis = 1)
    return result1

def warp_homography(source, target_shape, Hinv):
    """
    Warp the source image into the target coordinate frame using a provided
    inverse homography transformation.

    Args:
        source:         A 3-channel image represented as a numpy array.
        target_shape:   A 3-tuple indicating the desired results height, width, and channels, respectively
        Hinv:           A homography that maps locations in the result to locations in the source image.

    Returns:
        An image of target_shape with source's type containing the source image warped by the homography.
    """
    # TODO: allocation a numpy array of zeros that is size target_shape and the same type as source.
    h=target_shape[0]
    w=target_shape[1]
    ch=target_shape[2]
    h=int(h)
    w=int(w)
    target = np.zeros([h,w,ch],dtype='float32')

    # TODO: Iterate over all pixels in the target image
    sarr = np.empty([3, 1], dtype = 'float32')
    for x in range(w):
    	for y in range(h):
    		#px = np.array([x,y])
		#apply_homography is to be used but it was giving an axis error
		#thus, have used the formula here
            	# TODO: apply the homography to the x,y location
    		sarr[2,0] = Hinv[2,0]*x+Hinv[2,1]*y+Hinv[2,2]*1
    		sarr[0,0] = float((Hinv[0,0]*x+Hinv[0,1]*y+Hinv[0,2]*1)/sarr[2,0])
    		sarr[1,0] = float((Hinv[1,0]*x+Hinv[1,1]*y+Hinv[1,2]*1)/sarr[2,0])
		# TODO: check if the homography result is outside the source image. If so, move on to next pixel.
    		if (sarr[0,0]>=(source.shape[1]-1)) or (sarr[1,0]>=(source.shape[0]-1)) or (sarr[0,0]<0) or (sarr[1,0]<0):
    			continue
    		else:
		# TODO: Otherwise, set the pixel at this location to the bilinear interpolation result.
    			b = source[:,:,2]
    			g = source[:,:,1]
    			r = source[:,:,0]
    			value = bilinear_interp(r,(sarr[0,0],sarr[1,0]))
    			target[y,x,0] = value
    			value = bilinear_interp(g,(sarr[0,0],sarr[1,0]))
    			target[y,x,1] = value
    			value = bilinear_interp(b,(sarr[0,0],sarr[1,0]))
    			target[y,x,2] = value
    # return the output image
    return target


def rectify_image(image, source_points, target_points, crop):
    """
    Warps the input image source_points to the plane defined by target_points.

    Args:
        image:          The input image to warp.
        source_points:  The coordinates in the input image to warp from.
        target_points:  The coordinates to warp the corresponding source points to.
        crop:           If False, all pixels from the input image are shown. If true, the image is cropped to 
                        not show any black pixels.
    Returns:
        A new image containing the input image rectified to target_points.
    """

    # TODO: Compute the rectifying homography H that warps the source points to the target points.
    original_box = np.array([0,485,0,0,640,0,640,485]).reshape(4,2)
    H = compute_H(source_points,target_points)

    # TODO: Apply the homography to a rectangle of the bounding box of the of the image to find the warped bounding box in the rectified space.
    wbox = apply_homography(H,original_box)

    # Find the min_x and min_y values in the warped space to keep.
    if crop:
    	# TODO: pick the second smallest values of x and y in the warped bounding box
    	xm=np.amin(wbox[:,0])
    	ym=np.amin(wbox[:,1])
    	a=0
    	b=0
    	xarr=np.empty([3,1])
    	yarr=np.empty([3,1])
    	for i in range(4):
    		if wbox[i,0]>xm:
    			xarr[a,0]=wbox[i,0]
    			a=a+1
    		if wbox[i,1]>ym:
    			yarr[b,0]=wbox[i,1]
    			b=b+1
    	min_x=np.amin(xarr[:,0])
    	min_y=np.amin(yarr[:,0])
    else:
    	# TODO: Compute the min x and min y of the warped bounding box
    	min_x=np.amin(wbox[:,0])
    	min_y=np.amin(wbox[:,1])
    # TODO: Compute a translation matrix T such that min_x and min_y will go to zero
    T = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]])
    # TODO: Compute the rectified bounding box by applying the translation matrix to the warped bounding box.
    rbox= np.empty([4, 3], dtype = 'float32')
    for i in range(4):
    	rbox[i,0]=wbox[i,0]-min_x
    	rbox[i,1]=wbox[i,1]-min_y
    	rbox[i,2]=1
    # TODO: Compute the inverse homography that maps the rectified bounding box to the original bounding box
    #original_box = np.array([0,485,0,0,640,0,640,485]).reshape(4,2)
    Hinv = compute_H(rbox[:,0:2],original_box)

    # Determine the shape of the output image
    if crop:
        # TODO: Determine the side of the final output image as the second highest X and Y values of the rectified bounding box
        xa=np.amax(rbox[:,0])
        ya=np.amax(rbox[:,1])
        am=0
        bm=0
        xar=np.empty([3,1])
        yar=np.empty([3,1])
        for i in range(4):
                if rbox[i,0]<xa:
                        xar[am,0]=rbox[i,0]
                        am=am+1
                if rbox[i,1]<ya:
                        yar[bm,0]=rbox[i,1]
                        bm=bm+1
        max_x=np.amax(xar[:,0])
        max_y=np.amax(yar[:,0])
    else:
    	# TODO: Determine the side of the final output image as the maximum X and Y values of the rectified bounding box
    	max_x=np.amax(rbox[:,0])
    	max_y=np.amax(rbox[:,1])

    # TODO: Finally call warp_homography to rectify the image and return the result
    rectified_image = warp_homography(np.array(image),(max_y,max_x,3),Hinv)
    return rectified_image

def blend_with_mask(source, target, mask):
    """
    # Blends the source image with the target image according to the mask.
    # Pixels with value "1" are source pixels, "0" are target pixels, and
    # intermediate values are interpolated linearly between the two.

    # Args:
        # source:     The source image.
        # target:     The target image.
        # mask:       The mask to use

    # Returns:
        # A new image representing the linear combination of the mask (and it's inverse)
        # with source and target, respectively.
    # """

    # TODO: First, convert the mask image to be a floating point between 0 and 1
    #mask_arr = np.array(mask)
    max = np.max(mask)
    maskn = mask/max
    # TODO: Next, use it to make a linear combination of the pixels
    result = (1-maskn)*target + (maskn*source)

    # TODO: Convert the result to be the same type as source and return the result
    return result

def composite_image(source, target, source_pts, target_pts, mask):
    """
    # Composites a masked planar region of the source image onto a
    # corresponding planar region of the target image via homography warping.

    # Args:
        # source:     The source image to warp onto the target.
        # target:     The target image that the source image will be warped to.
        # source_pts: The coordinates on the source image.
        # target_pts: The corresponding coordinates on the target image.
        # mask:       A greyscale image representing the mast to use.
    # """
    # TODO: Compute the homography to warp points from the target to the source coordinate frame.
    H = compute_H(target_pts,source_pts)

    # TODO: Warp the source image to a new image (that has the same shape as target) using the homography.
    homo_source = warp_homography(np.array(source),(target.shape[0],target.shape[1],target.shape[2]),H)

    # TODO: Blend the warped images and return them.
    comp = blend_with_mask(homo_source,target,mask)
    return comp


def rectify(args):
    """
    The 'main' function for the rectify command.
    """

    # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # load the destination points, or select some smart default ones if None
    if args.dst == None:
        height = np.abs(
            np.max(source_points[:, 1]) - np.min(source_points[:, 1]))
        width = np.abs(
            np.max(source_points[:, 0]) - np.min(source_points[:, 0]))
        args.dst = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    target_points = np.array(args.dst).reshape(4, 2)

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Compute the rectified image
    result = rectify_image(inputImage, source_points, target_points, args.crop)
    #result=result.astype(uint8)
    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


def composite(args):
    # """
    # The 'main' function for the composite command.
    # """

    # # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # # load the target image
    logging.info('Loading target image %s' % (args.target))
    targetImage = load_image(args.target)

    # # load the mask image
    logging.info('Loading mask image %s' % (args.mask))
    maskImage = load_image(args.mask)

    # # If None, set the source points or sets them to the whole input image
    if args.source == None:
    	(height, width, _) = inputImage.shape
    	args.source = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    # # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # # Loads the target points into a 4-by-2 array
    target_points = np.array(args.dst).reshape(4, 2)

    # Compute the composite image
    result = composite_image(inputImage, targetImage,source_points, target_points, maskImage)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Warps an image by the computed homography between two rectangles.')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_rectify = subparsers.add_parser(
        'rectify', help='Rectifies an image such that the input rectangle is front-parallel.')
    parser_rectify.add_argument('input', type=str, help='The image to warp.')
    parser_rectify.add_argument('source', metavar='f', type=float, nargs=8,
                                help='A floating point value part of x1 y1 ... x4 y4')
    parser_rectify.add_argument(
        '--crop', help='If true, the result image is cropped.', action='store_true', default=False)
    parser_rectify.add_argument('--dst', metavar='x', type=float, nargs='+',
                                default=None, help='The four destination points in the output image.')
    parser_rectify.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_rectify.set_defaults(func=rectify)

    parser_composite=subparsers.add_parser('composite',help='Warps the input image onto the target points of the target image.')
    parser_composite.add_argument('input',type=str,help='The source image to warp.')
    parser_composite.add_argument('target',type=str,help='The target image to warp to.')
    parser_composite.add_argument('dst', metavar='f',type=float,nargs=8,help='A floating point value part of x1 y1 ... x4 y4 defining the box on the target image.')
    parser_composite.add_argument('mask', type=str, help='A mask image the same size as the target image.')
    parser_composite.add_argument('--source', metavar='x', type=float, nargs='+',
                                  default=None, help='The four source points in the input image. If ommited, the whole image is used.')
    parser_composite.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_composite.set_defaults(func=composite)
    args = parser.parse_args()
    args.func(args)
