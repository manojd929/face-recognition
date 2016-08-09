import cv2
import numpy as numpy

#we need a classifier model that can be used to detect faces in image 
#opencv provides an xml file that can be used for this purpose
fileHandle = open('csvgirish.txt','r')
lst = list()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
for eachline in fileHandle :
   	line = eachline.rstrip()
   	lst = line.split(';')
   	imagepath = lst[0]
   	labelpath = int(lst[1])
   	print 'imagepath = ',imagepath,' label path = ',labelpath
   	image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(image)
        print "faces detected : {} ",format(len(faces))
        for(x,y,w,h) in faces :
	    cv2.rectangle(image , (x,y) , (x+w,y+h) , (0,255,0) , 2)

        cv2.imshow('Faces Detected ',image)
        cv2.waitKey(2)
        cv2.destroyAllWindows()
