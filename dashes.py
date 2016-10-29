"""This code is specifically for the SOLID cartons. Objective is to find the dashes (-) in the Solid code
Assumes input images are roughly of size 3024x3024
Outputs an image that satisfies the requirements of tensorflow"""
import cv2
from PIL import Image
import numpy as np

# Overall objective: Find the dashes.
path = "output2.png"
img = cv2.imread(path) # read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
print "The original image size is " + str(img.shape)

# Step 1: Do thresholding
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] # thresholding works decently, but need to get rid of the false negatives and false positives
# Image.fromarray(gray).show()
print "New image shape is: " + str(gray.shape)
# Step 2: Look for contours in the shape of a dash
a, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # finds all the contours
img = cv2.drawContours(img, contours, -1, (0,255,0), 3)


count = 0
passed_conts = []
for cnt in contours: # next, will find the contours that are (approximately) dashes
    epsilon = 0.10*cv2.arcLength(cnt,True) # 0.10 is a param that perhaps needs calibration
    approx = cv2.approxPolyDP(cnt, epsilon, True) # approximation of shapes
    if len(approx) == 4: # if the contour has 4 edges (rectangles have 4 dashes, discard everything else)
        if not (cv2.contourArea(cnt) < 100.0 or abs(approx[0][0][0]-approx[1][0][0]) < 7 or abs(approx[0][0][1]-approx[3][0][1]) < 7 or
                        abs(approx[0][0][0]-approx[1][0][0]) < abs(approx[0][0][1]-approx[3][0][1]) or
                        abs(approx[0][0][1]-approx[2][0][1]) > abs(approx[0][0][0]-approx[2][0][0])):
            passed_conts.append([approx]) # collect the contours that don't violate any of the above
count = 0
passes = [] # contains the indexes of contours that pass the test
for x in range(0, len(passed_conts)-1): # check whether the the contours are roughly on the same horizontal line
    for y in range(x+1, len(passed_conts)):
        if abs(passed_conts[x][0][0][0][1] - passed_conts[y][0][0][0][1]) < 20:
            cv2.drawContours(img, passed_conts[x], -1, (255, 0, 0), 3)
            cv2.drawContours(img, passed_conts[y], -1, (255, 0, 0), 3)
            print "Contours passed and have been drawn onto the image "
            count += 1
            passes.append(x)
            passes.append(y)

Image.fromarray(img).show()

if count > 1:
    print "Check what went wrong...seems too many dashes were found"

background = np.zeros((28, 28), gray.dtype) # create black background

# Next, extract the dashes, place on 28x28.
# Slice the region (should be from the thresholded image)
for i in range(0, len(passes)):
    sli = gray[passed_conts[i][0][0][0][1]:passed_conts[i][0][3][0][1], passed_conts[i][0][0][0][0]:passed_conts[i][0][1][0][0]]
    # slicing works nicely
    sli = 255 - sli # invert the colors, want the dash to be white
    h = sli.shape[0]
    w = sli.shape[1]
    sli = Image.fromarray(sli)
    factor = 20.0/w # careful about datatypes
    print factor
    h_new = int(factor*h)
    if h_new == 0:
        h_new = 1
    res = sli.resize((20, h_new), Image.ANTIALIAS) # may want to use other kinds of interpolation when resizing
    res.show() # display the result
    res_array = np.asarray(res) # convert from Image to numpy array
    print res_array
    for k in range(0, len(res_array)):
        for j in range(0, len(res_array[k])):
            background[k+4][j+4] = res_array[k][j]
    Image.fromarray(background).show()
    # Now you can save them as image files or feed to tensorflow if you want.
