#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv


# In[2]:


# This uses Laptop Camera and Show you in new window
source=cv.VideoCapture(0)

win_name='camera preview'


while cv.waitKey(1)!=27: #The window will not close Until you hit Esc
    has_frame,frame=source.read()
    if not has_frame:
        break
    cv.imshow("camera preview",frame)

source.release()
cv.destroyWindow("camera preview")


# In[3]:


import numpy as np
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):#The window will not close Until you hit q
        break
#When everything done, release the capture
cap.release()
cv.destroyAllWindows()


# In[4]:


import cv2 as cv
import numpy as np
PREVIEW=0
BLUR=1
FEATURES=2
CANNY=3
feature_params=dict(maxCorners=500,qualityLevel=0.2,minDistance=15,blockSize=9)
[1]
image_filter=PREVIEW
alive=True
source=cv.VideoCapture(0)
win_name="window"

while alive:
    has_frame,frame=source.read()
    if not has_frame:
        break
    
    frame=cv.flip(frame,1)
    if image_filter==PREVIEW:
        result=frame
    elif image_filter==CANNY:
        result=cv.Canny(frame,80,150)
    elif image_filter==BLUR:
        result=cv.blur(frame,(5,5))
    elif image_filter==FEATURES:
        result=frame
        frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        corners=cv.goodFeaturesToTrack(frame_gray,**feature_params)
        if corners is not None:
            for x,y in np.float32(corners).reshape(-1,2):
                cv.circle(result,(int(x),int(y)),10,(0,255,0),1)
        
    cv.imshow("window",result)
    key=cv.waitKey(1)
    if key==27 or key==ord('q'): # will quit your window after hitting q
        alive=False
    elif key==ord('c'): # will go to canny mode (which is nothing but outline of you and surrounding) after hitting c
        image_filter=CANNY
    elif key==ord('f'): # will capture and show circle around your features using edges of features after hitting f
        image_filter=FEATURES
    elif key==ord('b'): # will go make your capture blur after hitting b
        image_filter=BLUR
    elif key==ord('p'): # original format after hitting p
        image_filter=PREVIEW
        
source.release()
cv.destroyAllWindows()


# In[15]:


#Face detection and extraction used on single image
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread("C:/Users/DELL/Downloads/IMG-20190508-WA0002.jpg")
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#colorc onversion is required
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3,7 ) # features can be changed if its not detecting face or detecting non facial parts as face
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    c=cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # cv.rectangle(image,(x,y) coordinates of top left and bottom right corners of rectangle(both necessary),color in(r,g,b),thickness)
    cropedimage=img[y:y+h,x:x+w] # creates cropped image
    cv2.imwrite('C:/Users/DELL/faces/'+str(w) + str(h) + '_faces.jpg',cropedimage) # save imaage in your desired location
    
# Display the output
width=500
height=500
dim=(width,height)
img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
cv2.imshow('img', img)
cv2.waitKey()


# In[57]:


img.shape


# In[ ]:


# face detection and extraction from folder containing multiple images
import cv2
import matplotlib.pyplot as plt
import glob
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') # for face detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml') # for eye detection
def detect(gray, frame) :
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = frame[y:y+h, x:x+w]

        cv2.imwrite("C:/Users/DELL/Desktop/faces/"+str(w)+str(h)+"sddcv2.jpg", roi_color) # location to save cropped images




    return frame
image_files=glob.glob("C:/Users/DELL/Desktop/phone/Telegram Images/*") #location to read files
for images in image_files:
frame = cv2.imread(images)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canvas = detect(gray, frame)






cv2.destroyAllWindows() # Close the window


# In[5]:


# Face Detection And extraction used on folder containing multiple folders of multiple images having and not having faces
import cv2
import numpy
import glob
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
def detect(gray, frame) :
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = frame[y:y+h, x:x+w]

        cv2.imwrite("C:/Users/DELL/faces/"+str(w)+str(h)+"cv2.jpg", roi_color)#folder path for saving images




    return frame

# for importing multiple images from multiple folders
folders = glob.glob('C:/Users/DELL/Downloads/lfw-funneled/lfw_funneled/*')#folder path for folder containing folders
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        imagenames_list.append(f)

        
for image in imagenames_list:
    frame=cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)


# In[1]:


#for continuous capture of images from webcam and detecting(+extraction) of faces
import cv2
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
alive=True
No=0
Capture=No
while alive:
    ret,frame=capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        roi_color = frame[y:y+h, x:x+w]

        cv2.imwrite("C:/Users/DELL/faces/"+str(w)+str(h)+"cv2.jpg", roi_color)
    cv2.imshow('Color',frame)
    if cv2.waitKey(1)==ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()


# In[4]:


#for capture of images from webcam and detecting(+extraction) of faces if pressed a spacific key

import cv2

cam = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        roi_color = frame[y:y+h, x:x+w]
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite("C:/Users/DELL/faces/"+str(w)+str(h)+"cv2.jpg", roi_color)#location for save
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


# In[ ]:




