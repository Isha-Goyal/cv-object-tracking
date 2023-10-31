import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from os import listdir

folder_dir = "cv-object-tracking/images_and_videos/medical_professionals/1/images/"
numImages = len(os.listdir(folder_dir))

# create array with all the file names. separate cases for 0-9 and >9 so that they always have two digits
# assumption: you have more than 9 images
imNames = [f"first_frame_0{i}.png" for i in range(0, 10)] + [f"first_frame_{i}.png" for i in range(10, numImages)]

# Initiate SIFT detector
sift = cv.SIFT_create()

# loop through and play images
for i in range(numImages - 1):
    img1 = cv.imread(folder_dir + imNames[i], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(folder_dir + imNames[i+1], cv.IMREAD_GRAYSCALE)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3),plt.show()
    
    # cv.imshow('image', img1)
    # cv.waitKey(40)


# next steps:
# - use ROI to pick the object after the first frame
# - only show the keypoints that fall within that region
# - show the progression of images with only those keypoints
# - maybe, find centroid
# - maybe, recalculate ROI in every frame

# cv.imshow('image', img1)
# cv.waitKey(0)

# img2 = cv.imread('cv-object-tracking/images_and_videos/medical_professionals/1/images/first_frame_01.png', cv.IMREAD_GRAYSCALE)


# find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)

# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.imshow(img3),plt.show()
