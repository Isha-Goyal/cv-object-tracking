# cv-object-tracking
Comp Robo F23




# What was the goal of your project? Since everyone is doing a different project, you will have to spend some time setting this context.
Our goal was open-endedly to gain familiarity with openCV and with computer vision, since neither of us had used either before. We framed this through object tracking and went through several small "tasks" that built on each other. For a final deliverable, we decided to use a dataset with videos from hospital security cameras to perform object tracking on healthcare workers. We ran our tests on this data, and have we have created object tracking algorithm that could be integrated into anautomated system that could potentially streamline hospital operations and monitor workflow, contributing to increased operational efficiency in healthcare settings.

## How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.
We began by splitting our working into lowe and higher level object detection. For the lower level track, we started with color-based object detection. For the high level decection exploration, we started with using the built in openCV tracking algorythms on single object tracking.

## Progression map of our learning timeline

### Numpy Inrange
To better understand the fundamentals of how color tracking algorithms work, after implementing it with openCV, we implemented it on top of numpy. The original implementation used openCV's inRange function, which filters an image based on whether a pixel has a value within a specified range that selects for certain colors. We rewrote inRange using numpy matrix operations. We viewed this as three stacked matrix 'masks' for blue, green, and red. Each one was essentially comprised of booleans that would turn values outside the specified range into zeros.

### Higher level single person tracking
For this model, we employ one of three advanced tracking methods—MIL, KCF, or CSRT—depending on the chosen configuration, each of which is designed to follow the object's position across a sequence of images. The MIL (Multiple Instance Learning) tracker balances between the robustness of the tracker and the computational efficiency, using a form of supervised learning that considers multiple 'instances' to update the model. The KCF (Kernelized Correlation Filters) tracker enhances efficiency by utilizing properties of circulant matrix to solve the ridge regression problem, leading to faster processing speeds. CSRT (Channel and Spatial Reliability Tracker) further improves accuracy by employing channel reliability and spatial reliability, hence providing precise tracking, particularly for objects that undergo significant scale or appearance changes, albeit with a potential trade-off in speed compared to KCF.

### SIFT person tracking
After experimenting with some of openCV's built-in object tracking algorithms like mean shifting, we decided to implement SIFT tracking on a set of images from a video a security camera took in a hospital showing medical professionals moving around.

#### On SIFT
SIFT stands for Scale-Invariant Feature Transform, and its main benefits are that it is both scale and rotation invariant. To start, algorithms detect extrema (such as corners, edges, and areas of high contrast) to pick out keypoints. These keypoints can be oriented to match the orientation of keypoints detected in the comparison image. This step is what makes SIFT rotation invariant. Then, each keypoint is given a descriptor, which encodes data about the values and contrasts of the pixels near the keypoint. Based on these descriptors, keypoints can be matched to one another. There are several ways to do that. One method is brute force matching which evaluates the similarity of keypoint descriptors across images using the ratio test, which quantifies the "distance" between a pair of descriptors.

### Multi Person Tracking
In the context of multi-person tracking within hospital settings, we instantiate an individual tracker for each staff member detected in the initial frame, utilizing our choice of the MIL, KCF, or CSRT algorithms. For each person, a unique tracking object is created, and a bounding box is manually selected or automatically determined around them. This is achieved through a loop that initializes a new tracker instance with the `cv2.Tracker<Type>_create()` function for each bounding box. Each tracker is associated with a specific individual and is responsible for updating the position of the bounding box as the person moves across the camera's field of view. The tracking algorithm accounts for changes in appearance and position, adjusting each bounding box in subsequent frames. This collection of trackers operates concurrently, with each tracker independently updating its state without interference from the others. This allows for the simultaneous tracking of multiple individuals, paving the way for sophisticated analysis of staff movement and resource allocation in efforts to increase efficiency in healthcare facilities.


# Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.

We faced a design decision regarding object tracking management. Initially, we considered using OpenCV's MultiTracker_create for its simplicity in handling multiple trackers. However we decided to implement a loop that created and updated each tracker separately:

    trackers = []
    for bbox in bboxes:
        tracker = create_tracker(tracker_type)
        ok = tracker.init(first_frame, bbox)
        if ok:
            trackers.append(tracker)

In this snippet, create_tracker is a function that returns a new tracker instance based on the specified tracker_type. This approach provided us with the flexibility to handle each tracker's initialization and update cycle with fine-grained control. By individually managing the trackers, we ensured compatibility with the current OpenCV version and maintained the stability of the overall system. Although this decision increased the complexity of our tracking management code, it allowed us to proceed without the risk of broader system impacts and dependency issues.


More Ideas:
- decision to do both base level implementations on numpy as well as built in algorithms from openCV, and to spend time building understanding from the ground up
- why the medical professionals dataset compared to the ones we were using before

# What if any challenges did you face along the way?

One of the significant challenges we faced was the absence of native support for multi-object selection in the Python interface of OpenCV. OpenCV's built-in function selectROI is designed for selecting a single object, and unfortunately, it does not support the selection of multiple objects simultaneously.

To overcome this limitation, we developed a custom interface that allowed us to select multiple regions of interest (ROIs) sequentially by invoking selectROI in a loop. After each selection, we temporarily stored the bounding box coordinates in a list, and the user could press a specific key to indicate the completion of the selection process for all objects. Here's a code snippet illustrating this solution:

    # Read the first image to initialize the trackers and get the frame size
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    if first_frame is None:
        print('Cannot read the first image')
        sys.exit()

    height, width, layers = first_frame.shape
    frame_rate = 10  # Or the actual frame rate of image sequence
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Initialize trackers and bounding boxes
    trackers = []
    bboxes = []
    colors = []

    # Use OpenCV's selectROI to manually select bounding boxes
    while True:
        bbox = cv2.selectROI('MultiTracker', first_frame)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press 'q' to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # 'q' is pressed
            break

This custom interface allowed us to manually select multiple objects in a frame, which was a key requirement for accurately initializing the tracking of hospital staff in the video data. This approach, while more cumbersome than a native multi-selection tool, proved to be a reliable workaround that did not require changes to the existing OpenCV installation.

The second major hurdle we encountered was managing the complexities of our Python environment. A significant portion of our project timeline was dedicated to managing virtual environments, adjusting Anaconda configurations, and repartitioning virtual machine virtual disks.

More Ideas:
- figuring out a direction for the project when we didn't know very much about comp. vision and the possibilities and how much time each part would take
- ~~environment stuff?~~

# What would you do to improve your project if you had more time?

With more time to enhance our project, one significant improvement would be integrating automatic object detection to streamline the tracking initialization process. Currently, the requirement for manual selection of regions of interest (ROIs) to track multiple objects is time-consuming and subject to human error. With automated detection, we could employ advanced machine learning models, such as Convolutional Neural Networks (CNNs), to accurately and efficiently identify individuals in the first frame of the video. This would not only expedite the setup phase by eliminating the need for manual ROI specification but also increase the system's scalability and robustness.

More Ideas:
- More sophisticated tracking algorithm that can work on harder datasets (e.g. the electric scooter one)
- Multi person tracking to a higher degree
- ~~Automatic object detection (don't need to specify the ROI manually)~~

# Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

Since I was not very confident at the start of this project, splitting it up to take ownership over a particular part was very helpful. I could sit there and take as much time with it as I needed and google as many "dumb" questions or refresh on basics as I needed without feeling pressured to go quickly or understand or keep up. That process gave me more confidence, both in my ability to use openCV and understand computer vision, but also to figure out programming issues by myself, since I'm often not in a position to do that.

Additionally, splitting the project into discrete, manageable segments facilitated a more organic and efficient workflow. We constructed the project in a series of distinct modules. This modular approach significantly enhanced our comprehension of each individual component, allowing for a deeper grasp of the system as a whole. It ensured that, at any pause point, we had multiple working parts in the project, showing clear progress and allowing us to adjust our targets easily. This incremental strategy proved to be more effective than pursuing a rigid, singular end-goal, allowing for adaptability and reassessment of our direction and methods.