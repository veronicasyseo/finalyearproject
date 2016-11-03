"""This code is specifically for the SOLID cartons. Objective is to find the dashes (-) in the Solid code
Assumes input images are roughly of size 3024x3024
Outputs an image that satisfies the requirements of tensorflow"""
import cv2
from PIL import Image
import numpy as np
from tesserocr import PyTessBaseAPI

# Overall objective: Find the dashes.
path = "output2.png"
img = cv2.imread(path) # read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
print "The original image size is " + str(img.shape)

# Step 1: Do thresholding
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] # thresholding works decently, but need to get rid of the false negatives and false positives
binary = gray.copy()
# Image.fromarray(gray).show()
print "New image shape is: " + str(gray.shape)
# Step 2: Look for contours in the shape of a dash
a, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # finds all the contours
# img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

boundingrects = [] # for storing the details of bounding rectangles

for cnt in contours: # get all the bounding rectangles of the contours
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 8 and h > 8 and 1000000 > (w*h) > 100: # eliminate the smallest boundingrects
        boundingrects.append((x, y, w, h))
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

# Image.fromarray(img).show()
print boundingrects

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


def placeonbg(array): #input should be array of an image. Should be resized
    background = np.zeros((28, 28), array.dtype) # create black background, new one for each time. maybe gray.dtype instead
    for k in range(0, len(array)):
        for j in range(0, len(array[k])):
            background[k+int(round((28-array.shape[0])/2))][j+int(round((28-array.shape[1])/2))] = array[k][j]
    return background

# Next, extract the dashes, place on 28x28.
# Slice the region (should be from the thresholded image)
for i in range(0, len(passes)):
    sli = gray[passed_conts[i][0][0][0][1]:passed_conts[i][0][3][0][1], passed_conts[i][0][0][0][0]:passed_conts[i][0][1][0][0]]
    # slicing works nicely
    binary[passed_conts[i][0][0][0][1]:passed_conts[i][0][3][0][1],
        passed_conts[i][0][0][0][0]:passed_conts[i][0][1][0][0]] = 255 # probably not necessary to remove dashes?
    sli = 255 - sli # invert the colors, want the dash to be white
    h = sli.shape[0]
    w = sli.shape[1]
    sli = Image.fromarray(sli)
    factor = 20.0/w # careful about datatypes
    print factor
    h_new = int(factor*h)

    if h_new == 0:
        h_new = 1
        print "I have a problem with heights"
    # resize to with 20, and appropriate height
    res = sli.resize((20, h_new), Image.ANTIALIAS) # may want to use other kinds of interpolation when resizing
    # res.show() # display the result
    res_array = np.asarray(res) # convert from Image to numpy array
    # place the dash centered on a black background
    output = placeonbg(res_array)
    #Image.fromarray(output).show()

digit_no = 1
# Can do factoring, instead of repeating code...
if len(passed_conts) == 2:
    with PyTessBaseAPI(psm = 10) as api:
        api.SetVariable("tessedit_char_whitelist", "0123456789")
        height = abs(passed_conts[passes[0]][0][0][0][1]-passed_conts[passes[0]][0][3][0][1])

        y_top = max(0, passed_conts[passes[0]][0][0][0][1]-5*height)
        y_bottom = min(passed_conts[passes[0]][0][0][0][1]+5*height, img.shape[1])

        region = binary[y_top:y_bottom, :] # This should (hopefully) contain the Solid code line
        Image.fromarray(region).show()

        # Next, want to get rid of the rectangles that are within other rectangles
        boundingrects.sort()
        print boundingrects
        boundingrects.pop(0) # assumes that the boundingrect of the entire region always will be found
        popper = []
        prev_outer = 0
        for a in range(0, len(boundingrects)-1): # compare a and a+1
            if boundingrects[a][0] + boundingrects[a][2] > boundingrects[a+1][0] + boundingrects[a+1][2]\
                    or boundingrects[prev_outer][0] + boundingrects[prev_outer][2] > boundingrects[a+1][0]+boundingrects[a+1][2]: # need to also consider y-dim
                popper.append(a+1) # need to deal with cases where two pop-able elements appear in sequence, to correctly find all pops
                prev_outer = a
        print len(popper)
        for b in range(0, len(popper)):
            boundingrects.pop(popper[len(popper)-b-1])
            # print "pop!"
        print boundingrects

        region1 = []
        region2 = []
        region3 = []

        keep_rects = []

        if len(passes) == 2:
            x_left_dash = min(passed_conts[passes[0]][0][0][0][0], passed_conts[passes[1]][0][0][0][0]) # assumes 2 dashes found
            x_right_dash = max(passed_conts[passes[0]][0][1][0][0], passed_conts[passes[1]][0][1][0][0]) # assumes 2 dashes found

            for rect in boundingrects:
                if not (rect[1] > y_bottom) or rect[1] < y_top: # reject the rectangles that go outside our region
                    if rect[0] < x_left_dash and rect[0] + rect[2] < x_left_dash:
                        region1.append(rect)
                    elif rect[0] > x_right_dash:
                        region3.append(rect)
                    elif x_left_dash < rect[0] and rect[0] + rect[2] < x_right_dash:
                        region2.append(rect)
                    else:
                        print "Not in any region "
            # Reason for splitting into regions is so that we can control the number of characters found (should be 2|3|3)
            if len(region2) < 3:
                print "Seems we are missing something from region2"
            for x in range(max(0,len(region1)-2), len(region1)):
                keep_rects.append(region1[x])
            for x in range(0, len(region2)):
                keep_rects.append(region2[x])
            for x in range(0, min(len(region3), 3)):
                keep_rects.append(region3[x])

            for r in keep_rects:
                x = r[0]
                y = r[1]
                w = r[2]
                h = r[3]
                sli = binary[y:y + h, x:x + w]
                sli = 255 - sli
                sli = Image.fromarray(sli)  # in order to be able to resize
                if w > h:  # wider than higher
                    factor = 20.0 / w  # careful with data types
                    h = int(factor * h)
                    w = 20
                else:  # height larger than width
                    factor = 20.0 / h
                    w = int(factor * w)
                    h = 20
                sli = sli.resize((w, h), Image.ANTIALIAS)
                sli = np.asarray(sli)
                on_bg = placeonbg(sli)
                focr = Image.fromarray(on_bg)
                # focr.show()
                path = "digit" + str(digit_no) + ".png"
                focr.save(path)
                api.SetImage(focr)
                print api.GetUTF8Text()
                digit_no += 1


else:
    print "Couldn't find the 2 dashes, GG"
    print len(passed_conts)
    sum_w = 0
    candidates = []

    for r in boundingrects:
        sum_w += r[3]
    avg_w = sum_w/len(boundingrects)

    for r in boundingrects:
        if r[3] < 0.5*avg_w and r[2] > 1.2*r[3]:  # needs calibration
            print "Candidate"
            candidates.append(r)
    region1 = []
    region2 = []
    region3 = []

    if len(candidates) == 2:
        with PyTessBaseAPI(psm = 10) as api:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            cv2.rectangle(img, (candidates[0][0], candidates[0][1]), (candidates[0][0]+candidates[0][2], candidates[0][1]+candidates[0][3]), (255, 0, 0), 3)
            cv2.rectangle(img, (candidates[1][0], candidates[1][1]), (candidates[1][0]+candidates[0][2], candidates[1][1]+candidates[1][3]), (255, 0, 0), 3)
            # good to go, so let's do some work
            # Step 1: Get the x-coords of the two dashes
            x_left_dash = min(candidates[0][0], candidates[1][0])
            x_right_dash = max(candidates[0][0] + candidates[0][2], candidates[1][0] + candidates[1][2])
            y_top = max(0, candidates[0][1] - 6 * candidates[0][3])
            y_bottom = candidates[0][1] + 6 * candidates[0][3]

            boundingrects.sort()
            print boundingrects
            boundingrects.pop(0)  # assumes that the boundingrect of the entire region always will be found
            popper = []
            prev_outer = 0
            for a in range(0, len(boundingrects) - 1):  # compare a and a+1
                if boundingrects[a][0] + boundingrects[a][2] > boundingrects[a + 1][0] + boundingrects[a + 1][2] \
                                or boundingrects[prev_outer][0] + boundingrects[prev_outer][2] > boundingrects[a + 1][0] + \
                                boundingrects[a + 1][2]:  # need to also consider y-dim
                    popper.append(a + 1)  # need to deal with cases where two pop-able elements appear in sequence, to correctly find all pops
                    prev_outer = a
            print len(popper)
            for b in range(0, len(popper)):
                print boundingrects.pop(popper[len(popper) - b - 1])
                print "pop!"
            print boundingrects

            for rect in boundingrects:
                if not (rect[1] > y_bottom or rect[1] < y_top):  # reject the rectangles that go outside our region
                    if rect[0] < x_left_dash and rect[0] + rect[2] < x_left_dash:
                        region1.append(rect)
                    elif rect[0] > x_right_dash:
                        region3.append(rect)
                    elif x_left_dash < rect[0] and rect[0] + rect[2] < x_right_dash:
                        region2.append(rect)
                    else:
                        print "Not in any region "
            keep_rects = []
            if len(region2) < 3:
                print "Seems we are missing something from region2"
            for x in range(max(0, len(region1) - 2), len(region1)):
                keep_rects.append(region1[x])
            for x in range(0, len(region2)):
                keep_rects.append(region2[x])
            for x in range(0, min(len(region3), 3)):
                keep_rects.append(region3[x])

            for r in keep_rects:
                x = r[0]
                y = r[1]
                w = r[2]
                h = r[3]
                sli = binary[y:y + h, x:x + w]
                sli = 255 - sli
                sli = Image.fromarray(sli)  # in order to be able to resize
                if w > h:  # wider than higher
                    factor = 20.0 / w  # careful with data types
                    h = int(factor * h)
                    w = 20
                else:  # height larger than width
                    factor = 20.0 / h
                    w = int(factor * w)
                    h = 20
                sli = sli.resize((w, h), Image.ANTIALIAS)
                sli = np.asarray(sli)
                on_bg = placeonbg(sli)
                focr = Image.fromarray(on_bg)
                # focr.show()
                api.SetImage(focr)
                print api.GetUTF8Text()
        # Split into 3 groups based on the x-coordinates, keeping in mind the y-coordinates.
    elif len(candidates) == 0 or 1:
        print "GG you die again. Perhaps your box is of assort type?"
    elif len(candidates) > 2:
        print "Oh, that's gonna be complicated. Will be implemented soon..."

# Draw in different colors
hmm = Image.fromarray(img)
# hmm.show()
# hmm.save("hmm.png")