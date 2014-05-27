#!/usr/bin/python
import socket
import cv2
import numpy
import rospy
import cv

TCP_IP = 'localhost'
TCP_PORT = 30000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()

capture = cv2.VideoCapture(0)

while not rospy.is_shutdown():
	ret, frame = capture.read()

	cv2.imwrite('abc.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
	#encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
	#result, imgencode = cv2.imencode('.jpg', frame, encode_param)
	#bitmap = cv.CreateImageHeader((imgencode.shape[1], imgencode.shape[0]), cv.IPL_DEPTH_8U, 3)
	#cv.SetData(bitmap, imgencode.tostring(),imgencode.dtype.itemsize * 3 * imgencode.shape[1])
	f = open('abc.jpg', "rb")
	rawImage = f.read()
	f.close()
	
	#data = numpy.array(imgencode)
	#stringData = bitmap.tostring()
	#print stringData
	#print type(bitmap)
	
#	conn.sendall( str(len(bitmap)).ljust(16));
	conn.sendall( rawImage );
sock.close()

decimg=cv2.imdecode(data,1)
#cv2.imshow('CLIENT',decimg)
#cv2.waitKey(0)
cv2.destroyAllWindows() 
