import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3

class BallTracker(Node):
    """ The BallTracker is a Python object that encompasses a ROS node 
        that can process images from the camera and search for a ball within.
        The node will issue motor commands to move forward while keeping
        the ball in the center of the camera's field of view. """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('ball_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.red_lower_bound = 0
        self.green_lower_bound = 0
        self.blue_lower_bound = 0
        self.red_upper_bound = 255
        self.green_upper_bound = 255
        self.blue_upper_bound = 255
        self.rbg_tuned_values = [13, 100, 26, 87, 255, 121] # rl, gl, bl, ru, gu, bu

        self.h_lower_bound = 0
        self.s_lower_bound = 0
        self.v_lower_bound = 0
        self.h_upper_bound = 179
        self.s_upper_bound = 255
        self.v_upper_bound = 255
        self.hsv_tuned_values = [0, 106, 21, 122, 236, 90] # hl, sl, vl, hu, su, vu

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window')
        cv2.namedWindow('binary_window')
        cv2.namedWindow('image_info')
        cv2.setMouseCallback('video_window', self.process_mouse_event)

        cv2.namedWindow('binary_image')
        cv2.createTrackbar('red lower bound', 'binary_window', self.red_lower_bound, 255, self.set_red_lower_bound)
        cv2.createTrackbar('green lower bound', 'binary_window', self.green_lower_bound, 255, self.set_green_lower_bound)
        cv2.createTrackbar('blue lower bound', 'binary_window', self.blue_lower_bound, 255, self.set_blue_lower_bound)
        cv2.createTrackbar('red upper bound', 'binary_window', self.red_upper_bound, 255, self.set_red_upper_bound)
        cv2.createTrackbar('green upper bound', 'binary_window', self.green_upper_bound, 255, self.set_green_upper_bound)
        cv2.createTrackbar('blue upper bound', 'binary_window', self.blue_upper_bound, 255, self.set_blue_upper_bound)

        cv2.namedWindow('hsv_window')
        cv2.createTrackbar('h lower bound', 'hsv_window', self.h_lower_bound, 179, self.set_h_lower_bound)
        cv2.createTrackbar('s lower bound', 'hsv_window', self.s_lower_bound, 255, self.set_s_lower_bound)
        cv2.createTrackbar('v lower bound', 'hsv_window', self.v_lower_bound, 255, self.set_v_lower_bound)
        cv2.createTrackbar('h upper bound', 'hsv_window', self.h_upper_bound, 179, self.set_h_upper_bound)
        cv2.createTrackbar('s upper bound', 'hsv_window', self.s_upper_bound, 255, self.set_s_upper_bound)
        cv2.createTrackbar('v upper bound', 'hsv_window', self.v_upper_bound, 255, self.set_v_upper_bound)

        cv2.namedWindow('my_test_window')

        while True:
            self.run_loop()
            time.sleep(0.1)


    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values
            associated with a particular pixel in the camera images """
        self.image_info_window = 255*np.ones((500,500,3))
        cv2.putText(self.image_info_window,
                    'Color (b=%d,g=%d,r=%d)' % (self.cv_image[y,x,0], self.cv_image[y,x,1], self.cv_image[y,x,2]),
                    (5,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))
        
    def myInRange(self, im, lower, upper):
        
        # create a test image that's black with a white square in the middle
        test = np.zeros((400, 400))
        
        for i in range(100, 300):
            for j in range(100, 300):
                test[i, j] = 1 

        layer1 = test # im(:, :, 0)
        layer1[layer1 < lower[2]] = 0
        layer1[layer1 > upper[2]] = 0
        # layer1[layer1 > 0] = 1

        #if you return test before the layer1 code, it'll return the right thing. If you return it after, it'll be all black.
        # also, with the normal video stream, if you keep everything but the line where you're turning nonzero values into 1s, then the picture does 
        # appear as grayscale. adding that line makes it all black. not the same as for test.

        return test
    


    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        print(self.cv_image)
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            self.binary_image = cv2.inRange(self.cv_image, (self.rbg_tuned_values[0], self.rbg_tuned_values[1], self.rbg_tuned_values[2]), (self.rbg_tuned_values[3], self.rbg_tuned_values[4], self.rbg_tuned_values[5])) 
            # to go back to sliders for rgb, put this in: (self.blue_lower_bound,self.green_lower_bound,self.red_lower_bound), (self.blue_upper_bound,self.green_upper_bound,self.red_upper_bound))
            # to go back to tuned rgb, put this in: (self.rbg_tuned_values[0], self.rbg_tuned_values[1], self.rbg_tuned_values[2]), (self.rbg_tuned_values[3], self.rbg_tuned_values[4], self.rbg_tuned_values[5])
            
            self.hsv_bin_image = cv2.inRange(self.cv_image, (self.hsv_tuned_values[0], self.hsv_tuned_values[1], self.hsv_tuned_values[2]), (self.hsv_tuned_values[3], self.hsv_tuned_values[4], self.hsv_tuned_values[5]))  
            # to go back to tuned hsv, put this in: (self.hsv_tuned_values[0], self.hsv_tuned_values[1], self.hsv_tuned_values[2]), (self.hsv_tuned_values[3], self.hsv_tuned_values[4], self.hsv_tuned_values[5]
            # to go back to slider hsv, put this in: (self.h_lower_bound, self.s_lower_bound, self.v_lower_bound), (self.h_upper_bound, self.s_upper_bound, self.v_upper_bound)
            
            cv2.imshow('binary_window', self.binary_image)
            cv2.imshow('hsv_window', self.hsv_bin_image)

            self.test_image = self.myInRange(self.cv_image, (30, 30, 30), (100, 100, 100))
            cv2.imshow('my_test_window', self.test_image)



            if hasattr(self, 'image_info_window'):
                cv2.imshow('image_info', self.image_info_window)
            cv2.waitKey(5)

    
    def set_red_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.red_lower_bound = val

    def set_green_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.green_lower_bound = val

    def set_blue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.blue_lower_bound = val

    def set_red_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.red_upper_bound = val

    def set_green_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.green_upper_bound = val

    def set_blue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.blue_upper_bound = val

    def set_h_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.h_lower_bound = val

    def set_s_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.s_lower_bound = val

    def set_v_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.v_lower_bound = val

    def set_h_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.h_upper_bound = val

    def set_s_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.s_upper_bound = val

    def set_v_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """

        self.v_upper_bound = val


def main(args=None):
    rclpy.init()
    n = BallTracker("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()