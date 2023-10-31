import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('cv-object-tracking/images_and_videos/medical_professionals/1/images/first_frame_00.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('cv-object-tracking/images_and_videos/medical_professionals/1/images/first_frame_01.png', cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()

# kp = sift.detect(gray,None)
# img=cv.drawKeypoints(gray,kp,img)
# cv.imwrite('first_frame_00_sift_keypoints.png',img)

# img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('first_frame_00_sift_keypoints.png',img)

# img = cv.imread('first_frame_00_sift_keypoints.png')
# cv.imshow('image', img)
# cv.waitKey(0)

# des = cv.compute(gray, kp)