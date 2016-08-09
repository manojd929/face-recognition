import PIL
from PIL import Image
import cv2,os
import numpy as np
from numpy import array
from socket import *
import numpy
from array import*
import pickle

# Set the socket parameters
host = "localhost"
port = 21567
buf = 4096
addr = (host,port)

# Create socket
UDPSock = socket(AF_INET,SOCK_DGRAM)
def get_images_and_labels():

    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    
    i=0
    
    fileHandle = open('csvldareducedtrain.txt','r')
    lst = list()
    
    for eachline in fileHandle :
    	line = eachline.rstrip()
    	lst = line.split(';')
    	imagepath = lst[0]
    	labelpath = int(lst[1])
    	print 'imagepath = ',imagepath,' label path = ',labelpath
    	image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
        faces = faceCascade.detectMultiScale(image)
        print "number of faces :",format(len(faces))
        for(x,y,w,h) in faces:
                cv2.namedWindow('training Images', cv2.WINDOW_NORMAL)
                cv2.imshow('training Images',image)
                cv2.waitKey(1)
        	images.append(image)
        	labels.append(labelpath)

    # return the images list and labels list
    return images, labels


# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
cv2.destroyAllWindows()
# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createFisherFaceRecognizer()


images, labels = get_images_and_labels() #Call above function


cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

attend = dict()

attend["1"] = "Girish"
attend["2"] = "Nethra"
attend["3"] = "Anvitha"
fileHandle = open('csvldareducedtest.txt','r')
lst = list()
    
for eachline in fileHandle :
    line = eachline.rstrip()
    lst = line.split(';')
    imagepath = lst[0]
    label = lst[1]
    print '\n\nimagepath = ',imagepath,' label = ',label
    predict_image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    
    #detect faces in the predict image
    faces = faceCascade.detectMultiScale(predict_image)
    print "Total number of faces detected in the given image : ",format(len(faces))
    
    attendance_list = list()
    i = int(format(len(faces)))
   
    for (x,y,w,h) in faces:
        #cv2.rectangle(predict_image , (x,y) , (x+w,y+h) , (0,255,0) , 2)
        cv2.namedWindow('Face Detected', cv2.WINDOW_NORMAL)
        cv2.imshow('Face Detected',predict_image)
	nbr_predicted,conf = recognizer.predict(predict_image)
        
        #convert nbr_predicted value to string type
        key = str(nbr_predicted)
        if label == key :
            print "Correctly Recognized"
        else :
            print "Incorrectly recognized"
	print "This Person is recognized as ",nbr_predicted," : ",attend[key],"with confidence ",conf
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #add it to the list 
        attendance_list.append(int(nbr_predicted))
   
        if(UDPSock.sendto(pickle.dumps(key),addr)):
            print "Sending message"
        cv2.destroyAllWindows()


if(UDPSock.sendto(pickle.dumps('End'),addr)):
    print ""        
UDPSock.close()
print "\n\nEnd"
cv2.destroyAllWindows()
