""" With automatic line extraction - basically a way to avoid too much pre-processing"""

import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI

path = "your path to input image"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print "Shape of grayscale image is:  " + str(gray.size)
gray = gray.astype('float32')
gray /= 255
dct = cv2.dct(gray)
vr = 1.  # vertical ratio
hr = .95  # horizontal
dct[0:vr * dct.shape[0], 0:hr * dct.shape[1]] = 0
gray = cv2.idct(dct)
gray = cv2.normalize(gray, -1, 0, 1, cv2.NORM_MINMAX)
gray *= 255
gray = gray.astype('uint8')

gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24)), # Originally (15,15)
                        iterations=1)
gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), # Originally (11,11)
                        iterations=1)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

temp = Image.fromarray(gray).show()

abac, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boxmask = np.zeros(gray.shape, gray.dtype)

i = 0

contlist = [] # Too be used for larger (merged) thresholds, may not be necessary

disthreshold = 10 # This is currently creating large bias towards the bottom-right corner of the image.
y_dim = img.shape[0]
x_dim = img.shape[1]

comparisons = 10 # not used currently?

def overlap(row1, row2): # Note: will have to check against contlist also, to see if overlap
    """Returns true if the rectangles overlap"""
    if row1[2]+0.3*disthreshold >= row2[0] and (row2[1] <= row1[3]+disthreshold or row2[1] <= row1[1]+disthreshold):
        # row1[2]+disthreshold >= row2[0] and (row2[1] <= row1[3]+disthreshold or row2[1] <= row1[1]+disthreshold)
        return True # check the above conditions...Part 1 Correct
    else:
        return False

def proximitycheck(matrix):
    """Returns a list of contours after merging close and overlapping contours"""
    output = []
    while len(matrix) > 0: # Keep doing until no more elements in the original matrix
        done = False # done is for the current i-value
        if len(output) == 0:
            output.append(matrix.pop(0))
        else: # Deal with row1 = row2?
            # Do comparison within the input matrix
            row1 = matrix[0] # To be compared with all the other elements until match is found.
             # Cause infinite loop?
            for j in range(1, len(matrix)): # Need to deal with out of bound- does not break out of loop properly...so the length of the array changes while iterating
                if not done and j < len(matrix): # avoid out of bound
                    row2 = matrix[j]
                    if overlap(row1, row2):
                        x_min = min((row1[1], row2[1]))
                        y_min = min((row1[0], row2[0]))
                        x_max = max((row1[3], row2[3]))
                        y_max = max((row1[2], row2[2]))
                        matrix.pop(j)
                        matrix.pop(0)
                        # Should check if overlap with any elements in the output matrix first # output.append(y_min, x_min, y_max, x_max)
                        row3 = [y_min, x_min, y_max, x_max]
                        # Start with len(output)-1 and iterate at most 5 times
                        print str(len(output))
                        if len(output) > 0:
                            print output
                        for l in range(0, len(output)): #Need to deal with out of bound issues
                            if done:
                                break
                            elif overlap(row3, output[max(len(output)-l-1, 0)]): #reduce cost later
                                r = max(len(output)-l-1, 0)
                                x_min = min(row3[1], output[r][1])
                                y_min = min(row3[0], output[r][0])
                                x_max = max(row3[3], output[r][3])
                                y_max = max(row3[2], output[r][2])
                                output.pop(r)
                                row4 = [y_min, x_min, y_max, x_max]
                                output.append(row4)
                                done = True

                        if not done: # if does not match any of the other elements in output matrix
                            output.append(row3)
                            done = True
            if not done:
                output.append(matrix.pop(0))
                done = True
                        # remember to break out of loop and to remove element from matrix

    return output
lower = 0.0001 # proably needs adjusting
upper = 0.80 # may need adjustments
scale = 1.08
with PyTessBaseAPI(psm=6) as api:
    print "Before checking area, the number of contours was: " + str(len(contours))
    for i in xrange(len(contours)):
        if upper > cv2.contourArea(contours[i]) / img.size > lower:  # Don't do for all contours, since smaller ones are from noise
            # may want to consider filtering out the larger INITIAL contours
            x, y, w, h = cv2.boundingRect(contours[i])
            x_left = int(x * (2-scale))
            x_right = int((x + w) * scale)
            y_top = int(y)
            y_bottom = int(y + h)
            contpts = [y_top, x_left, y_bottom, x_right]
            # contpts = [x_left, x_right, y_top, y_bottom]
            contlist.append(contpts)
            # if cv2.contourArea(contours[i]) / img.size > lower:
            # Note: may need to handle out of bounds problems
            # cv2.rectangle(boxmask, (x_left, y_top), (x_right, y_bottom), color=255,
                          # thickness=-1)  # thickness = -1 originally
            # res = img[y_top:y_bottom, x_left:x_right]
            # res = Image.fromarray(res)
            # api.SetImage(res)
            # print "This is from contour # " + str(i)
            # print api.GetUTF8Text()
            # path = "cont" + str(i)
            # res.save(path, "png")

    contlist.sort()
    print "The length of the original list is " + str(len(contlist))
    merged = proximitycheck(contlist)
    print "The length of the merged list is " + str(len(merged))

    # merged = proximitycheck(merged)
    # print "After merging once more, we have: " + str(len(merged))

    for cnt in merged:
        y_top = cnt[0]
        x_left = cnt[1]
        y_bottom = cnt[2]
        x_right = cnt[3]
        res = img[y_top:y_bottom, x_left:x_right]
        res = Image.fromarray(res)
        api.SetImage(res)
        print api.GetUTF8Text()

        cv2.rectangle(boxmask, (x_left, y_top), (x_right, y_bottom), color=255, thickness = -1) # originally thickness = -1
        # res = img[y_top:y_bottom, x_left:x_right]


            # Original code: cv2.rectangle(boxmask,(x,y),(x+w,y+h),color=255,thickness=-1)
# cv2.imshow('done',img&cv2.cvtColor(boxmask,cv2.COLOR_GRAY2BGR))
cv2.imwrite('output.png', img & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))

# Need to do some operations on the image before feeding it to OCR in order to improve accuracy:
ocr_img = cv2.imread('output.png')
ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
# Do Otsu binarization on the image:
ocr_otsu = cv2.threshold(ocr_gray, 0, 255, cv2.THRESH_OTSU)[1]
ocr_ready = Image.fromarray(ocr_otsu)
# cv2.waitKey(0)
# Finally, feed the whole processed image WITHOUT noise removal
with PyTessBaseAPI(psm = 6) as api:
    api.SetImage(ocr_ready)
    print "Feeding the processed - and slightly cleaned - image to tesseract gives: "
    print api.GetUTF8Text()
    img = api.GetThresholdedImage()
    img.show()
    print "Shape of thresholded image:  "
    print img.size
    # api.GetThresholdedImage().show()
    # print "The thresholded image has the following dimensions: " + str(api.GetThresholdedImage().shape)
    ls = api.GetTextlines()
    img = cv2.imread(path)
    for line in ls:
        x = line[1]['x']
        y = line[1]['y']
        w = line[1]['w']
        h = line[1]['h']
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    result = Image.fromarray(img)

    print "Shape of the original image: "
    print result.size
    result.show()

    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3

    # print api.GetBoxText()
    # api.GetThresholdedImage().show()
