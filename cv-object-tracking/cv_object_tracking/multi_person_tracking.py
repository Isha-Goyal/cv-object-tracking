import cv2
import sys
import os
from random import randint

# Check OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':
    # Set up tracker type
    tracker_types = ['MIL', 'KCF', 'CSRT']
    tracker_type = tracker_types[2]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video_path = 'tracked_output.mp4'

    # Path to the folder containing images
    image_folder = '/home/reuben/ros2_ws/src/mini_projects/cv-object-tracking/cv-object-tracking/images_and_videos/medical_professionals/2/images'

    # List of all images in the folder that end with '.png'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort the images by name

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
    #cv2.destroyAllWindows()

    # Create a tracker for each bounding box
    for bbox in bboxes:
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        tracker.init(first_frame, bbox)
        trackers.append(tracker)

    # Process each frame
    for image_name in images:
        frame = cv2.imread(os.path.join(image_folder, image_name))
        if frame is None:
            continue

        # Update each tracker and draw the bounding boxes
        for i, tracker in enumerate(trackers):
            ok, bbox = tracker.update(frame)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[i], 2)

        # Write the frame to the video
        video_writer.write(frame)

        # Display frame
        cv2.imshow('Tracking', frame)

        # Break out of the loop if the user presses 'ESC'
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # 'ESC' is pressed
            break

    # Release everything when finished
    video_writer.release()
    cv2.destroyAllWindows()
