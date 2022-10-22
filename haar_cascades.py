#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils.video import VideoStream
import imutils
import time
import cv2
import os


# In[2]:


detectorPaths = {
	"face": "haarcascade_frontalface_default.xml",
	"eyes": "haarcascade_eye.xml",
	"smile": "haarcascade_smile.xml",
}


# In[5]:


detect = cv2.CascadeClassifier()
detectors = {}


# In[8]:


detectors


# In[7]:


for name, path in detectorPaths.items():
    paths = os.path.sep.join([os.path.abspath(os.getcwd()), path])
    detectors[name] = cv2.CascadeClassifier(paths)


# In[11]:


vs = VideoStream(src=0).start()
time.sleep(2)


# In[12]:


while True:
    frame = vs.read()
    
    # frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = detectors['face'].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (fX, fY, fW, fH) in faceRects:
        faceROI = gray[fX:fX + fH, fY:fY + fW]
        eyeRects = detectors['eyes'].detectMultiScale(
            faceROI, scaleFactor=1.1, minNeighbors=10,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE
        )

        smileRects = detectors['smile'].detectMultiScale(
            faceROI, scaleFactor=1.1, minNeighbors=10,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (eX, eY, eW, eH) in eyeRects:
            ptA = (fX + eX, fY + eY)
            ptB = (fX + eX + eW, fY + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
        
        for (sX, sY, sW, sH) in smileRects:
            ptA = (sX + sX, sY + sY)
            ptB = (sX + sX + sW, sY + sY + sH)
            cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)

        cv2.rectangle(frame, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()

