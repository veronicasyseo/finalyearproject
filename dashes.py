"""This code is specifically for the SOLID cartons. Objective is to find the dashes (-) in the Solid code
Assumes input images are roughly of size 3024x3024"""
import cv2
from PIL import Image

# Overall objective: Find the dashes.
path = "path to output file from extractandmergelines.py, should contain the Solid code"
img = cv2.imread(path) # read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

# Step 1: Do thresholding
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] # thresholding works decently, but need to get rid of the false negatives and false positives
# Image.fromarray(gray).show()

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

for x in range(0, len(passed_conts)-1): # check whether the the contours are roughly on the same horizontal line
    for y in range(x+1, len(passed_conts)):
        if abs(passed_conts[x][0][0][0][1] - passed_conts[y][0][0][0][1]) < 20:
            cv2.drawContours(img, passed_conts[x], -1, (255, 0, 0), 3)
            cv2.drawContours(img, passed_conts[y], -1, (255, 0, 0), 3)
            print "Contours passed and have been drawn onto the image "
Image.fromarray(img).show()


