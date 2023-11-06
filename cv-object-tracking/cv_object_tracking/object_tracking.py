import cv2
import sys
import os

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':
    # Set up tracker
    tracker_types = ['MIL', 'KCF', 'CSRT']
    tracker_type = tracker_types[2]

    if tracker_type == 'MIL':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Path to the folder containing images
    image_folder = '/home/reuben/ros2_ws/src/mini_projects/cv-object-tracking/cv-object-tracking/images_and_videos/medical_professionals/2/images'

    # List of all images in the folder that end with '.png'.
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort the images by name (or any other method required)

    # Read the first image to initialize the tracker
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    if first_frame is None:
        print('Cannot read the first image')
        sys.exit()

    # Define an initial bounding box or use cv2.selectROI to select the box
    bbox = cv2.selectROI(first_frame, False)

    # Initialize tracker with the first frame and bounding box
    ok = tracker.init(first_frame, bbox)

    # Loop over all the images in the folder
    for image_name in images:
        frame = cv2.imread(os.path.join(image_folder, image_name))

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            print(p2)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Break from the loop if 'ESC' is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
