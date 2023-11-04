# cv-object-tracking
Comp Robo F23



# What was the goal of your project? Since everyone is doing a different project, you will have to spend some time setting this context.
# How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.
Our goal was open-endedly to gain familiarity with openCV and with computer vision, since neither of us had used either before. We framed this through object tracking and went through several small "tasks" that built on each other. We started with color-based object detection.

## Progression map of our learning timeline

### Numpy Inrange
To better understand the fundamentals of how color tracking algorithms work, after implementing it with openCV, we implemented it on top of numpy. The original implementation used openCV's inRange function, which filters an image based on whether a pixel has a value within a specified range that selects for certain colors. We rewrote inRange using numpy matrix operations. We viewed this as three stacked matrix 'masks' for blue, green, and red. Each one was essentially comprised of booleans that would turn values outside the specified range into zeros.

### Higher level single person tracking

### SIFT person tracking
After experimenting with some of openCV's built-in object tracking algorithms like mean shifting, we decided to implement SIFT tracking on a set of images from a video a security camera took in a hospital showing medical professionals moving around.

#### On SIFT
SIFT stands for Scale-Invariant Feature Transform, and its main benefits are that it is both scale and rotation invariant. To start, algorithms detect extrema (such as corners, edges, and areas of high contrast) to pick out keypoints. These keypoints can be oriented to match the orientation of keypoints detected in the comparison image. This step is what makes SIFT rotation invariant. Then, each keypoint is given a descriptor, which encodes data about the values and contrasts of the pixels near the keypoint. Based on these descriptors, keypoints can be matched to one another. There are several ways to do that. One method is brute force matching which evaluates the similarity of keypoint descriptors across images using the ratio test, which quantifies the "distance" between a pair of descriptors.

### Multi Person Tracking



# Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.

Ideas:
- decision to do both base level implementations on numpy as well as built in algorithms from openCV, and to spend time building understanding from the ground up
- why the medical professionals dataset compared to the ones we were using before

# What if any challenges did you face along the way?

Ideas:
- figuring out a direction for the project when we didn't know very much about comp. vision and the possibilities and how much time each part would take
- environment stuff?

# What would you do to improve your project if you had more time?

Ideas:
- More sophisticated tracking algorithm that can work on harder datasets (e.g. the electric scooter one)
- Multi person tracking to a higher degree
- Automatic object detection (don't need to specify the ROI manually)

# Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

Since I was not very confident at the start of this project, splitting it up to take ownership over a particular part was very helpful. I could sit there and take as much time with it as I needed and google as many "dumb" questions or refresh on basics as I needed without feeling pressured to go quickly or understand or keep up. That process gave me more confidence, both in my ability to use openCV and understand computer vision, but also to figure out programming issues by myself, since I'm often not in a position to do that.
