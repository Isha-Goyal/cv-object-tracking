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
r = cv.selectROI("select the area", img1) # [top left x, top left y, width, height]

x = int(r[0])
y = int(r[1])
width = int(r[2])
height = int(r[3])

for im_idx in range(len(imNames)):

    # pick img1 roi and crop image accordingly
    img1_roi = img1[y:y+height, x:x+width]

    # meant to protect against the roi being too close to the edges of the frame so the larger roi tries to index parts of the image that aren't there
    newY = int(y-height/2)
    newX = int(x-width/2)
    newHeight = int(2*height)
    newWidth = int(2*width)

    if newY <= 0:
        newY = 0
    elif newX <= 0:
        newX = 0


    img2_roi = img2[newY:newY + newHeight, newX:newX + newWidth]


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
    # img3 = cv.drawMatchesKnn(img1_roi,kp1,img2_roi,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    sumX, sumY = 0, 0

    # plt.imshow(img3),plt.show()

    for i in range(0, len(matches)):

        img1_roi = cv.circle(img1_roi, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), radius=1, color=(255, 255, 255), thickness=0)

        index = matches[i][1].trainIdx
        ptX = int(kp2[index].pt[0])
        ptY = int(kp2[index].pt[1])
        img2_roi = cv.circle(img2_roi, (ptX, ptY), radius=2, color=(255, 255, 255), thickness=1)

        # determine "center of mass" of identified points in second image
        sumX = sumX + ptX
        sumY = sumY + ptY

    # note: adding the original x and y of the roi helps keep all the point values relative to the original image rather
    # than the cropped image, which will be useful later when tracking the ROI across frames
    shiftX = x * len(matches)
    shiftY = y * len(matches)
    avgX = int(sumX/len(matches))
    avgY = int(sumY/len(matches))
    comX = int((sumX+shiftX)/len(matches))
    comY = int((sumY+shiftY)/len(matches))

    # draw center of mass on second image
    img2_roi = cv.circle(img2_roi, (avgX, avgY), radius=4, color=(255, 255, 255), thickness=3)

    cv.imshow('marked_image', img1_roi)
    cv.imshow('second_image', img2_roi)

    # img2 = cv.circle(img2, (comX, comY), radius=4, color=(255, 255, 255), thickness=3)
    cv.imshow('full_image', img2)

    cv.waitKey(0)

    # read next set of images
    img1 = img2
    img2 = cv.imread(folder_dir + imNames[im_idx+1], cv.IMREAD_GRAYSCALE)

    # assumption: keeping height and width of ROI same as the original
    x = int(comX - (width/2))
    y = int(comY - (height/2))
