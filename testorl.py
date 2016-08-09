import cv2,os
import cv
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
    
    fileHandle = open('/home/mano/Desktop/project_attendance/orl/orl_train_1.txt','r')
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

#attend = dict()
#attend["3"] = "Anvitha"
#attend["1"] = "Girish"
#attend["2"] = "Nethra"
fileHandle = open('/home/mano/Desktop/project_attendance/orl/orl_test_1.txt','r')
lst = list()
i = 0  
j = 0 
facescountdet = 0 
for eachline in fileHandle :
    line = eachline.rstrip()
    lst = list()
    lst = line.split(';')
    imagepath = lst[0]
    label     = lst[1]
    print '\n\nimagepath = ',imagepath,'label = ',label
    predict_image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    
    #detect faces in the predict image
    faces = faceCascade.detectMultiScale(predict_image)
    print "Total number of faces detected in the given image : ",format(len(faces))
    if (  ( int ( format ( len(faces) ) ) ) >= 1 ) :
        facescountdet += 1
    
    #cv2.namedWindow('Test Image',cv2.WINDOW_NORMAL)
    #cv2.imshow('Test Image',predict_image)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #attendance_list = list()
    
   
    for (x,y,w,h) in faces:
	nbr_predicted, conf = recognizer.predict(predict_image)
        
        #convert nbr_predicted value to string type
        key = str(nbr_predicted)
	print "This Person is recognized as ",nbr_predicted," with confidence :",conf
        if label == key:
            i +=1
            print "Correctly recognized : ",i
        else:
            j+=1
            print "Incorrect"
        #add it to the list 
        #attendance_list.append(int(nbr_predicted))
   
        cv2.namedWindow('Face Recognized', cv2.WINDOW_NORMAL)
        cv2.rectangle(predict_image , (x,y) , (x+w,y+h) , (0,255,0) , 2)
        cv2.imshow('Face Recognized ',predict_image)
        cv2.waitKey(50) 
        #label = int(nbr_predicted)
        #label_update.append(label)
        #recognizer.update(predict_image[y:y+h,x:x+w],np.array(label_update[j]))
        #listlength = len(attendance_list)
       
        #if i == listlength :
        #    print attendance_list
        #    if(UDPSock.sendto(pickle.dumps(attendance_list),addr)):
        #        print "Sending message"
        cv2.destroyAllWindows()

print "\n\nTotal Correctly recognized ",i
print "Total Incorrect recognized ",j
print "No. of faces detected          ",facescountdet
UDPSock.close()
cv2.destroyAllWindows()
