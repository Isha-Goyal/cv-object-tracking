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

image = cv.imread(folder_dir + imNames[0], cv.IMREAD_GRAYSCALE)
r = cv.selectROI("select the area", image) # [top left x, top left y, width, height]


# loop through and play images
# for i in range(numImages - 1):
img1 = cv.imread(folder_dir + imNames[0], cv.IMREAD_GRAYSCALE)
img2 = cv.imread(folder_dir + imNames[1], cv.IMREAD_GRAYSCALE)

x = r[0]
y = r[1]
width = r[2]
height = r[3]

img1_roi = img1[int(y):int(y+height),  
                    int(x):int(x+width)]
img2_roi = img2[int(y)-int(height/2):int(y+(2*height)), int(x)-int(width/2):int(x+width)]


# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_roi,None)
kp2, des2 = sift.detectAndCompute(img2_roi,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# # Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1_roi,kp1,img2_roi,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# plt.imshow(img3),plt.show()

for i in range(0, len(matches)):

    img1_roi = cv.circle(img1_roi, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), radius=1, color=(255, 255, 255), thickness=0)

    index = matches[i][1].trainIdx
    img2_roi = cv.circle(img2_roi, (int(kp2[index].pt[0]), int(kp2[index].pt[1])), radius=1, color=(255, 255, 255), thickness=0)

cv.imshow('marked_image', img1_roi)
cv.imshow('second_image', img2_roi)

cv.waitKey(0)

##############################################3
# cv.imshow('image', img2)
# cv.waitKey(0)

    
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

# def click_event(event, x, y, flags, params):
#    if event == cv.EVENT_LBUTTONDOWN:
#       print(f'({x},{y})')
      
#       # put coordinates as text on the image
#       cv.putText(img1_roi, f'({x},{y})',(x,y),
#       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
#       # draw point on the image
#       cv.circle(img1_roi, (x,y), 3, (0,255,255), -1)
 

# # create a window
# cv.namedWindow('Point Coordinates')

# # bind the callback function to window
# cv.setMouseCallback('Point Coordinates', click_event)
