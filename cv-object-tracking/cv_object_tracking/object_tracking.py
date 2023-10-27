import cv2
import sys
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    # Set up tracker.
    tracker_types = ['MIL','KCF', 'CSRT']
    tracker_type = tracker_types[2]

    if tracker_type == 'MIL':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        print(f"Tracker type {tracker_type} is not recognized. Exiting.")
        sys.exit()

 
    # Read video
    video = cv2.VideoCapture('/home/reuben/ros2_ws/src/mini_projects/cv-object-tracking/cv-object-tracking/images_and_videos/street.mp4')
 
    # Exit if video not opened.
    if not video.isOpened():
        print('Could not open video')
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
     
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Promt to define custom bounding box
    bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            print("Tracking Failure")

        # Result
        cv2.imshow("Tracking", frame)
 
        # Exit on ESC keystroke
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break