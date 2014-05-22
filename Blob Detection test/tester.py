#!/usr/bin/env python
import roslib
import sys
import rospy
import cv
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import time


capture=cv.CaptureFromCAM(0)

def talker():
    global capture
    bridge=CvBridge()
    pub = rospy.Publisher('chatter', Image)
    rospy.init_node('talker', anonymous=True)
    while not rospy.is_shutdown():
        frame=cv.QueryFrame(capture)
        image=frame;
        image=np.asarray(image[:,:])
        medianB=np.median(image[:,:,0]);
        medianG=np.median(image[:,:,1]);
        medianR=np.median(image[:,:,2]);
        #print median
        diffB=128-medianB
        diffG=128-medianG
        diffR=128-medianR
        image[:,:,0]=image[:,:,0]+diffB
        image[:,:,1]=image[:,:,1]+diffG
        image[:,:,2]=image[:,:,2]+diffR
        image[image>=255]=254
        image[image<=0]=1
        #print maxi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([50,150,0])
        upper_green = np.array([150,255,255])
        lower_blue = np.array([80,50,0])
        upper_blue = np.array([160,255,255])
        lower_red=np.array([0,160,0])
        upper_red=np.array([30,255,255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        mask3 = cv2.inRange(hsv, lower_red, upper_red)
        #a=image.shape
        #print a
        '''for i in range(1,a[1]):
            for j in range(1,a[2]):
                temp=image[i,j]
                temp=temp+diff
                print type(temp)
                if(temp<=255 and temp>=0):
                    image[i,j]=temp
        '''
    
        
        maxi=np.min(image)
        res = cv2.bitwise_and(image,image, mask= mask)
        res2 = cv2.bitwise_and(image,image, mask= mask2)
        res3 = cv2.bitwise_and(image,image, mask= mask3)
        final=res+res2+res3
        frame=cv.fromarray(final)
        #frame=final
        
        pub.publish(bridge.cv_to_imgmsg(frame, "bgr8"))


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
