""" With automatic line extraction - basically a way to avoid too much pre-processing"""

import cv2 # opencv-python
import numpy as np # numpy
from PIL import Image # Pillow
from tesserocr import PyTessBaseAPI # tesserocr


img = cv2.imread("path to image") # open image
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert to greyscale
gray = gray.astype('float32')
gray/=255
dct=cv2.dct(gray)
vr=1.#vertical ratio
hr=.95#horizontal
dct[0:vr*dct.shape[0],0:hr*dct.shape[1]]=0
gray=cv2.idct(dct)
gray=cv2.normalize(gray,-1,0,1,cv2.NORM_MINMAX)
gray*=255
gray=gray.astype('uint8')

gray=cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),
    iterations=1)
gray=cv2.morphologyEx(gray, cv2.MORPH_DILATE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)),
    iterations=1)
gray=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)[1] # Otsu is local thresholding, but is not too slow here

abac, contours,hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
boxmask=np.zeros(gray.shape,gray.dtype)

lower = 0.0001 # may need calibration

i = 0

with PyTessBaseAPI(psm=6) as api: # psm=6 will make tesseract assume a single uniform block of text

    for i in xrange(len(contours)):
        if cv2.contourArea(contours[i]) / img.size > lower: # Don't do for all contours, since smaller ones are from noise
            x,y,w,h = cv2.boundingRect(contours[i])

            # Expand the boundaries to improve likelihood of including relevant text
            x_left = int(x*0.95)
            x_right = int((x+w)*1.05)
            y_top = int(y*0.97)
            y_bottom = int((y+h)*1.03)

            # place the contours back onto a black background
            cv2.rectangle(boxmask,(x_left,y_top),(x_right,y_bottom),color=255,thickness=-1)

            # extract contours one by one and perform OCR on each of them
            res = img[y_top:y_bottom, x_left:x_right]
            res = Image.fromarray(res)

            api.SetImage(res)

            print "This is from contour # " + str(i)
            print api.GetUTF8Text()
            # path = "cont" + str(i)
            # res.save(path, "png")
            i += 1
            # Original code: cv2.rectangle(boxmask,(x,y),(x+w,y+h),color=255,thickness=-1)
# cv2.imshow('done',img&cv2.cvtColor(boxmask,cv2.COLOR_GRAY2BGR))

cv2.imwrite('output path',img&cv2.cvtColor(boxmask,cv2.COLOR_GRAY2BGR))

