import cv2
import numpy as np
import imageio

stitcher = cv2.Stitcher.create()
first = cv2.imread("ii1_cylindrical.jpg")
second = cv2.imread("ii2_cylindrical.jpg")
third = cv2.imread("ii3_cylindrical.jpg")
four = cv2.imread("ii4_cylindrical.jpg")
five = cv2.imread("ii5_cylindrical.jpg")
six = cv2.imread("ii6_cylindrical.jpg")
seven = cv2.imread("ii7_cylindrical.jpg")
result = stitcher.stitch((first, third))
# #,four,five,six))
blur = cv2.GaussianBlur(result[1],(3,3),0)

imageio.imwrite("panorama13.png", result[1])
