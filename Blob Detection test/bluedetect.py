import cv2
import numpy as np
import cv

#cv.NamedWindow("w1",cv.CV_WINDOW_AUTOSIZE)
#capture = cv.CaptureFromCAM(0)
image=cv2.imread('/home/niranjan/Desktop/Rock-Colors.JPG')
#img=np.asaray(image[:,:])
#while(1):
    #image=cv.QueryFrame(capture)
    #cv.ShowImage("w1",image)

image=np.asarray(image[:,:])
    

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([50,150,0])
upper_green = np.array([150,255,255])
lower_blue = np.array([80,50,120])
upper_blue = np.array([160,255,255])
lower_red=np.array([0,160,100])
upper_red=np.array([30,255,255])


mask = cv2.inRange(hsv, lower_green, upper_green)
mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
mask3 = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(image,image, mask= mask)
res2 = cv2.bitwise_and(image,image, mask= mask2)
res3 = cv2.bitwise_and(image,image, mask= mask3)
final=res+res2+res3
cv2.imshow('My Image',image)
cv2.imshow('Original',final)
    #pint final
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


#while(1):

    # Take each frame
    #_, frame = cap.read()
##    frame=cv.QueryFrame(capture)
##    img=np.asarray(frame[:,:])
##    img
##
##    # Convert BGR to HSV
##    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
##    
##    # define range of blue color in HSV
##    lower_blue = np.array([110,50,50])
##    upper_blue = np.array([130,255,255])
##
##    # Threshold the HSV image to get only blue colors
##    mask = cv2.inRange(hsv, lower_blue, upper_blue)
##
##    # Bitwise-AND mask and original image
##    res = cv2.bitwise_and(img,img, mask= mask
##    #dst = cv2.fastNlMeansDenoisingColored(res,None,10,10,7,21)
##    cv2.imshow('img',img)
##    cv2.imshow('mask',mask)
##    cv2.imshow('res',res)
##    k = cv2.waitKey(5) & 0xFF
##    if k == 27:
##        break

##cv2.destroyAllWindows()
