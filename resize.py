import PIL
import cv2
from PIL import Image

fileHandle = open('/home/mano/Desktop/1.txt','r')
lst = list()
i=0
for eachline in fileHandle :
    line = eachline.rstrip()
    lst = line.split(';')
    imagepath = lst[0]
    #labelpath = int(lst[1])
    print 'imagepath = ',imagepath,' label path = '#,labelpath
    image = Image.open(imagepath)
    image = image.resize((1000,1000),PIL.Image.ANTIALIAS)
    i+=1
    image.save('test.'+str(i)+'.jpeg')
    
    

