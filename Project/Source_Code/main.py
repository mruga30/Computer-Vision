import numpy as np
import imageio
from scipy.integrate import quad
from PIL import Image
import cv2
import sys
import argparse



def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def extract_corneal_reflection(I, bwImage, cx, cy, window_width):
    """I = input image
      notes [sx sy] = start point for starburst algorithm
       window_width = threshold for pupil edge detection"""

    if window_width % 2 == 0:
        print("Window width should be odd!")
        return None
    height = I.shape[0]
    width = I.shape[1]

    r = int(np.floor((window_width - 1) / 2))
    sx = int(np.max([np.round(cx - r), 1]))
    ex = int(np.min([np.round(cx + r), width]))
    sy = int(np.max([np.round(cy - r), 1]))
    ey = int(np.min([np.round(cy + r), height]))
    Iw = bwImage[sx:ex, sy:ey]
    Iw2 = I[sx:ex, sy:ey]

    imageio.imwrite('IW.jpg', Iw)
    threshold = 120
    x, y, = 0, 0
    score = [0]
    indx = 1

    for i in range(threshold, 0, -1):
        ret, Iwt = cv2.threshold(Iw, i, 255, cv2.THRESH_BINARY)
        output = cv2.connectedComponentsWithStats(Iwt, 8, cv2.CV_32S)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        centroids = output[3]
        print "Stats", stats.shape
        areas = stats[:, 4]
        print("Areas", areas)
        max_area_index = np.where(areas == np.amax(areas))[0][0]
        print("Max area index", max_area_index)
        deno = (np.sum(areas) - areas[max_area_index]) + 0.00000000000001
        score.insert(indx, float(areas[max_area_index]) / float(deno))
        print "Score", score
        print(indx)
        if score[indx] - score[indx - 1] < 0:
            x = centroids[max_area_index][0]
            y = centroids[max_area_index][1]
            ar = (np.sqrt(4.0 * areas[max_area_index] / np.pi)) / 2.0
            print("x:", x, "y:", y, "radius:", ar)
            break
        indx += 1

    x += sx - 1
    y += sy - 1
    fRadius = radius_calculation(ar, x, y, I)
    finalRadius = fRadius * 2.0
    print("Final Radius: ", finalRadius)
    img = clip_reflection(Iw2, x, y, finalRadius)
    imageio.imwrite('extracted_output.jpg', img)
    return x, y, finalRadius


def radius_calculation(initial_radius, xc, yc, I):
    px = 1
    r = initial_radius
    prevradius = 0
    while (abs(r - prevradius) > 0.99):
        prevradius = r
        a = quad(intensity, 0, 2*np.pi, args=(r + px, xc, yc, I))[0]
        b = quad(intensity, 0, 2*np.pi, args=(r - px, xc, yc, I))[0]
        r = r-a/b
    finalRadius = r
    return finalRadius


def intensity(theta, rad, xc, yc, I):
    x = xc + rad * np.cos(theta)
    y = yc + rad * np.sin(theta)
    value = I[int(x), int(y)]
    return value[0]


def clip_reflection(I, xc, yc, r):
    height = I.shape[0]
    width = I.shape[1]
    sx = int(np.max([np.round(xc - r), 1]))
    ex = int(np.min([np.round(xc + r), width]))
    sy = int(np.max([np.round(yc - r), 1]))
    ey = int(np.min([np.round(yc + r), height]))
    img = I[sx:ex, sy:ey]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--i', help='input image', required=True)
    parser.add_argument('-x', '--x', help='x center of window', required=True)
    parser.add_argument('-y', '--y', help='y center of window', required=True)
    parser.add_argument('-w', '--w', help='width of window', required=True)
    args = vars(parser.parse_args())
    colorImage = load_image(args['i'])
    bwImage = cv2.imread(args['i'])
    gray_image = cv2.cvtColor(bwImage, cv2.COLOR_BGR2GRAY)
    extract_corneal_reflection(colorImage, gray_image, args['y'], args['x'], args['w'])


