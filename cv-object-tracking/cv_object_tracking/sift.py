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

img1 = cv.imread(folder_dir + imNames[0], cv.IMREAD_GRAYSCALE)
img2 = cv.imread(folder_dir + imNames[1], cv.IMREAD_GRAYSCALE)
r = cv.selectROI("select the area", image) # [top left x, top left y, width, height]

x = int(r[0])
y = int(r[1])
width = int(r[2])
height = int(r[3])

for i in range(len(imNames)):
    
    # pick img1 roi and crop image accordingly
    img1_roi = img1[y:y+height, x:x+width]

    # meant to protect against the roi being too close to the edges of the frame so the larger roi tries to index parts of the image that aren't there
    newY = int(y-height/2)
    newX = int(x-width/2)
    newHeight = int(newY+(2*height))
    newWidth = int(newX+(2*width))

    if newY <= 0:
        newY = 0
    elif newX <= 0:
        newX = 0


    img2_roi = img2[newY:newHeight, newX:newWidth]


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
    sumX, sumY = 0, 0

    # plt.imshow(img3),plt.show()

    for i in range(0, len(matches)):

        img1_roi = cv.circle(img1_roi, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), radius=1, color=(255, 255, 255), thickness=0)

        index = matches[i][1].trainIdx
        x = int(kp2[index].pt[0])
        y = int(kp2[index].pt[1])
        img2_roi = cv.circle(img2_roi, (x, y), radius=2, color=(255, 255, 255), thickness=1)

        # determine "center of mass" of identified points in second image
        sumX = sumX + x
        sumY = sumY + y

    sumX = int(sumX/len(matches))
    sumY = int(sumY/len(matches))

    # draw center of mass on second image
    img2_roi = cv.circle(img2_roi, (sumX, sumY), radius=4, color=(255, 255, 255), thickness=3)

    cv.imshow('marked_image', img1_roi)
    cv.imshow('second_image', img2_roi)

    cv.waitKey(40)

    # read next set of images
    img1 = cv.imread(folder_dir + imNames[i], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(folder_dir + imNames[i+1], cv.IMREAD_GRAYSCALE)

    # to do: process so we get an ROI centered on the com
    # x = y = 
    # assumption: keeping height and width of ROI same as the original


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
