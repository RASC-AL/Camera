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


#capture=cv.CaptureFromCAM(1)

def talker():
    global capture
    bridge=CvBridge()
    pub = rospy.Publisher('chatter', Image)
    rospy.init_node('talker', anonymous=True)
    while not rospy.is_shutdown():
        capture=cv.CaptureFromCAM(1)
        
        frame=cv.QueryFrame(capture)
        #image=cv2.imread('/home/niranjan/Desktop/Rock-Colors.JPG')
        capture=cv.CaptureFromCAM(2)
        
        frame1=cv.QueryFrame(capture)

        image=frame;
        image=np.asarray(image[:,:])
        image1=frame1;
        image1=np.asarray(image1[:,:])
        both=np.hstack((img,img1))
        bitmap = cv.CreateImageHeader((both.shape[1], both.shape[0]), cv.IPL_DEPTH_8U, 3)
        cv.SetData(bitmap, both.tostring(), both.dtype.itemsize * 3 * both.shape[1])
        frame=bitmap
        image=frame
        '''medianB=np.median(image[:,:,0]);
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
        image[image<=0]=1'''
        #print maxi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([80,80,0])
        upper_blue = np.array([130,255,255])
        lower_green = np.array([50,80,0])
        upper_green = np.array([79,255,255])
        lower_red=np.array([0,160,0])
        upper_red=np.array([40,255,255])

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

    
        
        #maxi=np.min(image)
        res = cv2.bitwise_and(image,image, mask= mask)
        im1=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        contours1, hierarchy1 = cv2.findContours(im1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours1:
            x1,y1,w1,h1 = cv2.boundingRect(i)
            cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
        res2 = cv2.bitwise_and(image,image, mask= mask2)
        im2=cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
        contours2, hierarchy2 = cv2.findContours(im2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours2:
            x2,y2,w2,h2 = cv2.boundingRect(i)
            cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        res3 = cv2.bitwise_and(image,image, mask= mask3)
        im3=cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)
        contours3, hierarchy3 = cv2.findContours(im3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours3:
            x3,y3,w3,h3 = cv2.boundingRect(i)
            cv2.rectangle(image,(x3,y3),(x3+w3,y3+h3),(0,0,255),2)
	#mask8 = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
	#image[mask==0]=0
        #final=res+res3+res2
        #final=res+res2
	final=image
        #frame1=cv.fromarray(final)
        frame1=final
        #im2=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        #im3=cv2.cvtColor(im2,cv2.COLOR_GRAY2BGR)
        #ret,thresh=cv2.threshold(im2,127,255,0)
        
        #contours, hierarchy = cv2.findContours(im2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #for i in contours:
            #x,y,w,h = cv2.boundingRect(i)
            #cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        #frame1=im3
        frame1=cv.fromarray(frame1)
        #cv2.imshow('Features', im)       
        #frame=fin
	#res = cv2.bitwise_and(image,image, mask= mask3)
	#mask3=cv2.cvtColor(mask3,cv2.COLOR_GRAY2RGB)
	#frame=cv.fromarray(mask3)
        
	
        
        pub.publish(bridge.cv_to_imgmsg(frame1, "bgr8"))


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
