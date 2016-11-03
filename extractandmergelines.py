""" With automatic line extraction - basically a way to avoid too much pre-processing"""

import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, iterate_level
import csv
import Levenshtein

path = "path to your image file"
ocr_output_path = "output.txt"
asn_input_path = "path to ASN in csv format"

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

temp = Image.fromarray(gray)
temp.show()
temp.save("temp.png")

abac, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boxmask = np.zeros(gray.shape, gray.dtype)

i = 0

contlist = [] # To be used for larger (merged) contours

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

def proximitycheck(matrix): # Current problem: There are some pictures getting stuck..Resolved
    """Returns a list of contours after merging close and overlapping contours.
    May need special case for when the matrix (input) has only one element"""
    output = []
    while len(matrix) > 0: # Keep doing until no more elements in the original matrix
        done = False # done is for the current i-value
        if len(output) == 0:
            output.append(matrix.pop(0))
        elif len(matrix) == 1:
            # check if can match to any of the elements in the output array, otherwise simply append it
            row1 = matrix[0]
            row2 = output[len(output)-1] # the last element. Need error handling (possibly)
            if overlap(row1, row2):
                x_min = min((row1[1], row2[1]))
                y_min = min((row1[0], row2[0]))
                x_max = max((row1[3], row2[3]))
                y_max = max((row1[2], row2[2]))
                matrix.pop(0)
                row4 = [y_min, x_min, y_max, x_max]
                output.append(row4)
        else:
            # Do comparison within the input matrix
            row1 = matrix[0] # To be compared with all the other elements until match is found.

            for j in range(0, len(matrix)): # Need to deal with out of bound- does not break out of loop properly...so the length of the array changes while iterating
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
                            elif overlap(row3, output[max(len(output)-l-1, 0)]):
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
                if not done: # CHECK: Indenting
                    output.append(matrix.pop(0))
                    done = True

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
            # cv2.rectangle(boxmask, (x_left, y_top), (x_right, y_bottom), color=255,
                          # thickness=-1)  # thickness = -1 originally
            # res = img[y_top:y_bottom, x_left:x_right]
            # print "This is from contour # " + str(i)

    contlist.sort()
    print "The length of the original list is " + str(len(contlist))
    merged = proximitycheck(contlist)
    print "The length of the merged list is " + str(len(merged))

    for cnt in merged:
        y_top = cnt[0]
        x_left = cnt[1]
        y_bottom = cnt[2]
        x_right = cnt[3]
        res = img[y_top:y_bottom, x_left:x_right]
        res = Image.fromarray(res) #check if necessary
        # api.SetImage(res)
        # print api.GetUTF8Text() # Need to identify whether line 1 is scanned, and if yes,use this information to help improve accuracy later

        cv2.rectangle(boxmask, (x_left, y_top), (x_right, y_bottom), color=255, thickness = -1) # originally thickness = -1

cv2.imwrite('output.png', img & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))

# Need to do some operations on the image before feeding it to OCR in order to improve accuracy:
ocr_img = cv2.imread('output.png')
ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)

# Do Otsu binarization on the image:
ocr_otsu = cv2.threshold(ocr_gray, 0, 255, cv2.THRESH_OTSU)[1]
ocr_ready = Image.fromarray(ocr_otsu)

# Finally, feed the whole processed image
with PyTessBaseAPI(psm = 6) as api:
    api.SetImage(ocr_ready)
    api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
    print "Feeding the processed image to tesseract gives: "
    output_text = api.GetUTF8Text().encode("utf-8") # Encode as utf-8, otherwise would be ascii by python's default
    print output_text
    f = open("output.txt", 'w')
    f.write(output_text)
    img = api.GetThresholdedImage()
    img.show()
    print "Shape of thresholded image:  "
    print img.size
    # api.GetThresholdedImage().show()
    # print "The thresholded image has the following dimensions: " + str(api.GetThresholdedImage().shape)
    ls = api.GetTextlines()
    img = cv2.imread(path)

    for line in ls: # draw all the rectangles
        x = line[1]['x']
        y = line[1]['y']
        w = line[1]['w']
        h = line[1]['h']
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    result = Image.fromarray(img)

    print "Shape of the original image: "
    print result.size
    # result.show()
    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3

    # print api.GetBoxText()
    # api.GetThresholdedImage().show()
    iterator = api.GetIterator()
    iterator.Begin()
    level = RIL.SYMBOL
    for r in iterate_level(iterator, level):
        # print r.BoundingBox(level)
        x = r.BoundingBox(level)[0]
        y = r.BoundingBox(level)[1]
        x_2 = r.BoundingBox(level)[2]
        y_2 = r.BoundingBox(level)[3]

        img = cv2.rectangle(img, (x, y), (x_2, y_2), (0, 255, 0), 3) # Draw a green rectangle around each character found by OCR

    out = Image.fromarray(img)
    out.show()
    out.save("out.png")
    f.close()
    # Need to kill iterator to clear memory====

    # Want to show the bounding box of L1 of the SKU:
    ocr_data = []
    asn_data = []

    # Read the advance shipment notice for the purpose of checking OCR output against the feasible values
    with open(asn_input_path, "rb") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';',
                                quotechar='|')  # Spamreader is an object..need to save the info somehow

        for row in spamreader:  # Row only contains 1 row at any given time
            #print ', '.join(row)
            asn_data.append(row)
    csvfile.close()

    # Read the ocr output into a list
    with open(ocr_output_path) as f:
        ocr_data.append(f.readlines())
    f.close()

    smallest_lev = 1000
    index = 0
    # Currently shifts the bounding rectangle of L1 of SKU one line down
    blanks = 0
    for line in ocr_data[0]:
        if len(line) == 0:
            print "Soemhow have 0 length of line? "
        elif len(line) == 1:
            print "Blank line!"
            blanks += 1
        elif len(line) > 10:  # don't reject if is long enough to be L1 of SKU. May want to remove white-spaces from Line
            for i in range(1, len(asn_data)): # asn_data[0] is just the titles
                d = Levenshtein.distance(asn_data[i][3], line)
                if d < smallest_lev:
                    smallest_lev = d
                    smallest_pos = i
                    pos = index # sometimes pos turns out to be the wrong value..? Ends up referring to the line below instead of L1 of SKU.
        index += 1
    if blanks > 0:
        pos -= blanks-1
    print "Smallest Lev distance: " + str(smallest_lev) # Note: Whitespaces don't affect the pick of the best fit, but they do affect the given Levenshtein distance (score)
    if smallest_lev == 1:  # Since newline character is ignored
        print "Perfect match!"
    print "Found at position: " + str(smallest_pos)
    print "Has value: " + asn_data[smallest_pos][3]  # The corresponding value of L1 SKU.

    # The following are the details of L1 of SKU:
    x = ls[pos][1]['x']
    #print x
    y = ls[pos][1]['y']
    w = ls[pos][1]['w']
    #print w
    h = ls[pos][1]['h']
    im = cv2.imread(path)
    im_hw = im.copy()
    im = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3) # Bounding rectangle of the first line of SKU
    out = Image.fromarray(im)
    out.show()# look at that lovely blue rectangle!
    out.save("out2.png")
    c = 20 # constant, needs fine-tuning
    x_min = x-c # initial guess
    for oh in range(1, 4): # try the next 3 lines. Assumes there are at least 3 more lines after L1 of SKU
        if (oh+pos) < len(ls): # check out of bounds
            if ls[oh+pos][1]['x'] < x_min:
                x_min = ls[oh+pos][1]['x']
    x_hw1 = x_min - c
    y_hw1 = y + h
    x_hw2 = x + w
    #print x_hw2
    y_hw2 = y + int(5.0*h) # needs calibration
    # The above relies on correctly finding the first line of SKU AND correctly cropping the image based on line detection
    handwriting = cv2.imread(path)
    gray = cv2.cvtColor(handwriting, cv2.COLOR_BGR2GRAY)
    boxmask = np.zeros(gray.shape, gray.dtype)
    cv2.rectangle(boxmask, (x_hw1, y_hw1), (x_hw2, y_hw2), color=255, thickness=-1) # coords, not distances for 2nd part
    # cv2.imwrite('output2.png', img & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('output2.png', im_hw & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))

    # cv2.imwrite('output.png', img & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """To add: Automatic line rotation, in case some, but not all, of the printed lines are at a large angle.
     Also, need to find ways to improve the probability of correctly finding line 1 (e.g. 341-172235(63-03)"""
