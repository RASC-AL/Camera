#!/usr/bin/env python
import roslib
roslib.load_manifest('cam_test')
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


height=str(640)
width=str(480)
fps=str(30)
cam="5"
capture=cv.CaptureFromCAM(0)
capture1=None
sub=None
frame=None
check_fps_set=0

def callback1(ros_data):
    #pub = rospy.Publisher('/chatter', Image)
    #bridge=CvBridge()
    #print 'received image data of type:%s'% ros_data.format
    global sub
    global frame
    print "here"
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    frame=cv.fromarray(image_np)
    #sub.unregister()
    
def callback(data):
    print "data: " + data.data
    global height
    global width
    global capture
    global capture1
    global fps
    global cam
    global check_fps_set
    global sub
    string=str(data.data)
    b=string.split(',')
    height=str(b[2])
    width=str(b[3])
    cam=str(b[0])
    fps=str(b[1])
    if cam==str(0):
        capture=None
        capture1=None
        capture=cv.CaptureFromCAM(0)
        sub.unregister()
    if cam==str(1):
        capture=None
        capture1=None
        capture=cv.CaptureFromCAM(1)
        sub.unregister()
    if cam==str(2):
        capture=None
        capture1=None
        capture=cv.CaptureFromCAM(2)
        sub.unregister()
    if cam==str(3):
        capture=None
        capture1=None
        capture=cv.CaptureFromCAM(3)
        sub.unregister()
    if cam==str(5):
        capture=None
        capture1=None
        sub=rospy.Subscriber('/image_raw/compressed', CompressedImage,callback1,queue_size=1)
    if cam==str(4):
        capture=None
        capture1=None
        sub.unregister()
        capture=cv.CaptureFromCAM(0)
        capture1=cv.CaptureFromCAM(1)
        check_fps_set=1
        return
    check_fps_set=1
    #print "prev_cam: "+str(prev_cam)+" new cam: "+str(cam)
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,int(width))
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,int(height))
        
def talker():
    #cv.NamedWindow("")
    global height
    global width
    global fps
    global check_fps_set
    global capture
    global capture1
    global cam
    global sub
    global frame
    #capture =cv.CaptureFromCAM(0)
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,int(width))
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,int(height))
    #cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FPS,2)
    #rate=cv.GetCaptureProperty(capture,cv.CV_CAP_PROP_FPS)
    #print rate
    bridge=CvBridge()
    pub = rospy.Publisher('chatter', Image)
    rospy.init_node('talker', anonymous=True)
    
    rospy.Subscriber("config", String, callback)
    #rospy.Subscriber('/image_raw/compressed', CompressedImage, callback1,queue_size=1)
    r = rospy.Rate(int(fps)) # 10hz
    time1=0
    frames=0
    start_time=0
    check=0
    cv.QueryFrame(capture)
    frame=cv.QueryFrame(capture)
    if cam==str(5):
        sub=rospy.Subscriber('/image_raw/compressed', CompressedImage,callback1,queue_size=1)
    while not rospy.is_shutdown():
        #str = "hello world %s"%rospy.get_time()
        if cam!=str(5):
            cv.QueryFrame(capture)
            cv.QueryFrame(capture)
            cv.QueryFrame(capture)
            frame=cv.QueryFrame(capture)
        time1=time.time()
        #print "after frame capture: "+str(time1)
        if cam==str(4):
            frame=cv.QueryFrame(capture)
            frame1=cv.QueryFrame(capture1)
            img=np.asarray(frame[:,:])
            img1=np.asarray(frame1[:,:])
            both=np.hstack((img,img1))
            bitmap = cv.CreateImageHeader((both.shape[1], both.shape[0]), cv.IPL_DEPTH_8U, 3)
            cv.SetData(bitmap, both.tostring(), both.dtype.itemsize * 3 * both.shape[1])
            frame=bitmap
        if cam==str(5):
            #print "inside here"
            #sub=rospy.Subscriber('/image_raw/compressed', CompressedImage,callback1,queue_size=1)
            #print frame
        if check==0:
            check=1
            start_time=time.time()
        if check_fps_set==1:
            r = rospy.Rate(int(fps))
            print "fps: " + fps
            start_time=time.time()
            frames=0
            check_fps_set=0
        frames=frames+1
        if frames%10==0:
            curtime=time.time()
            diff=curtime-start_time
            fps=frames/diff
            print fps
        #ret,frame=capture.read()
        #image=np.asarray(frame[:,:])
        #a=image.shape
        #print a
        #rospy.loginfo(st)
        pub.publish(bridge.cv_to_imgmsg(frame, "bgr8"))
        time1=time.time()
        #print "after frame publish: "+str(time1)
        r.sleep()
        time1=time.time()
        #print "after sleep: "+str(time1)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
