import cv2
# from PIL import Image
import numpy as np
import math
#import matplotlib

#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt


def loadKNN():
    path = "/Users/sigurdandersberg/PycharmProjects/proj1/knn_data_large.npz"
    with np.load(path) as data:
        print data.files # list the files stored
        train = data['train'].astype(np.float32)
        train_labels = data['train_labels'].astype(np.float32)

    # do accuracy tests (should give 100%, but do it for sake of testing PT)
    # Carry out training
    knn = cv2.ml.KNearest_create()  # check later
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)  # need to get correct input...

    return knn

def deskew(img):
    """Deskew digit

	Parameters
	----------
	img : np.array
        2D digit array

	Returns
	-------
    dst : Deskewed digit
	"""
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    rot_mat = np.float32([[1, skew, -0.5*max(img.shape[0], img.shape[1])*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, rot_mat, (img.shape[0], img.shape[1]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def trim_padding(img):
    """Trim zeros rows and columns

	Parameters
	----------
	img : np.array
        2D digit array

	Returns
	-------
    dst : trimmed digit
	"""
    mask_row = np.all(np.equal(img, 0), axis=1)
    dst = img[~mask_row]

    mask_col = np.all(np.equal(dst, 0), axis=0)
    dst = dst[:, ~mask_col]

    return dst

def resize_with_constant_ratio(img, char_dim):
    """Resize image while keeping aspect ratio. Max dim is char_dim
	pad_dim is applied in order to have derivative friendly image

	Parameters
	----------
	img : np.array
        2D digit array
    char_dim : int
        dst dim

	Returns
	-------
    dst : resized digit
	"""
    roi_h = img.shape[0]
    roi_w = img.shape[1]

    max_dim = max(roi_w, roi_h)
    pad_dim = 2
    scale = float(char_dim-pad_dim) / max_dim
    if roi_w >= roi_h:
        new_w = int(char_dim-pad_dim)
        new_h = int(roi_h * scale)
    else:
        new_w = int(roi_w * scale)
        new_h = int(char_dim-pad_dim)

    dst = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return dst

def pad_digit(img, char_dim):
    """Pad zeros in order to get a square char_dimxchar_dim image

	Parameters
	----------
	img : np.array
        2D digit array
    char_dim : int
        image dim

	Returns
	-------
    dst : padded digit
	"""
    pad_h = char_dim-img.shape[0]
    pad_w = char_dim-img.shape[1]
    pad_h_b = math.floor(pad_h/2)
    pad_h_t = pad_h - pad_h_b
    pad_w_r = math.floor(pad_w/2)
    pad_w_l = pad_w - pad_w_r

    dst = np.hstack(( img, np.zeros((img.shape[0], pad_w_r))))
    dst = np.hstack(( np.zeros((dst.shape[0], pad_w_l)), dst))

    dst = np.vstack(( dst, np.zeros((pad_h_b, dst.shape[1]))))
    dst = np.vstack(( np.zeros((pad_h_t, dst.shape[1])), dst))

    return dst

class PatternComponent():
    def __init__(self, outer, width):
        """Structure of outer: x_min, points, x_max, outer_inner"""
        self.category = -1  # categories are 0, 1, 2 for not connecting, maybe connecting and definitely connecting. -1 would indicate an error
        self.outer = outer
        self.inner = []
        self.upper = []
        self.lower = []
        self.cutting_points_lower = []
        self.cutting_points_upper = []
        self.upper_left = []
        self.upper_right = []
        self.lower_left = []
        self.lower_right = []
        self.left = []
        self.right = []
        self.left_full = []
        self.right_full = []  # may be able to obtain by simply using + operator between lists ("concatinate"). But need to first divide into left and right hand side
        self.symbol_guesses = []
        self.symbol_distances = []
        self.beta = -1

        self.setBeta(width)  # always do when initialize then pattern component

    def setBeta(self, width):
        """Sets the beta value, which will later be used to give text outputs
        Values and interpretation: 0 - nothing weird. 1 - either small contour or close to edge. 2 - both small and close to edge.
        Here edge refers to LHS or RHS, not top/bottom. """
        self.beta = 0
        # need measure of contour size.
        contour_size = cv2.contourArea(self.outer[1])  # uses Green's formula to approximate the contour area
        # need a measure of the avg contour size (area)
        x_min = self.outer[0]
        x_max = self.outer[2]

        lower_bound = int(width*0.15)
        upper_bound = int(width*0.85)

        if x_min < lower_bound and contour_size > 450:
            self.beta = 1
        elif x_max > upper_bound and contour_size > 450:
            self.beta = 1
        elif contour_size < 450:
            self.beta = 1
        elif x_min < lower_bound:
            self.beta = 2
        elif x_max > upper_bound:
            self.beta = 2
        else:
            self.beta = 0

    def getBeta(self):
        return self.beta

    def setSymbolDistances(self, dist):
        if type(min(dist)) is int:
            self.symbol_distances.append(min(dist))
        else:
            self.symbol_distances.append(min(dist)[0
                                         ])

    def getSymbolDistances(self):
        return self.symbol_distances

    def getMinSymbol(self):
        sym = np.asarray(self.getSymbolGuesses())
        dis = np.asarray(self.getSymbolDistances())

        # get the index of the minimum distance, then use it to return the best guess symbol
        min_dist_index = np.argmin(dis)  # assume unique
        return sym[min_dist_index]

    def setSymbolGuesses(self, string):
        self.symbol_guesses.append(string)

    def getSymbolGuesses(self):
        """Unordered"""
        return self.symbol_guesses

    def getRankedSymbolGuesses(self):
        """Ranked"""
        guesses = self.getSymbolGuesses()
        distances = self.getSymbolDistances()

        dist_guess = zip(distances, guesses)  # zip so that we can rank them in order of increasing distance

        dist_guess = sorted(dist_guess, key=lambda x: x[0])

        distances, guesses = zip(*dist_guess)

        return guesses

    def findBoundingBox(self, contour):
        """Returns the corners of the bounding box in form of:
        x_min, x_max, y_min, y_max"""
        contour = contour
        x_min = 10000
        x_max = 0
        y_min = 10000
        y_max = 0

        # contour should be a long collection of points
        # find min and max values along each of axes 0 and 1.

        # could iterate through individual points, but might be slow?

        for point in contour:
            x = point[0]
            y = point[1]

            if x < x_min:
                x_min = x
            elif x > x_max:
                x_max = x

            if y < y_min:
                y_min = y
            elif y > y_max:
                y_max = y

        return x_min, x_max, y_min, y_max

    def getInnerContours(self):  # can be used to correctly draw the contours. But fails when want to get the points at a later point?
        inner_all = self.getInner()
        inner_contour_points = []
        if len(inner_all) > 0:
            for contour in inner_all:
                inner_contour_points += list(contour[1])

        return inner_contour_points

    def setLeftRightFull(self, c_b):  # again assume that cut is a straight vertical line
        self.left_full = []
        self.right_full = []  # reset for every single time we do a different cut

        x_val = c_b[0][0]
        outer = self.getOuter()
        for point in outer:
            if point[0] < x_val:
                self.left_full.append(point)
            elif point[0] > x_val:
                self.right_full.append(point)
            elif point[0] == x_val:
                self.right_full.append(point)
                self.left_full.append(point)

        inner = self.getInnerContours()

        if len(inner) > 0:
            for point in inner:  # may need to handle cases where there a multiple inner contours
                if point[0] < x_val:
                    self.left_full.append(point)
                elif point[0] > x_val:
                    self.right_full.append(point)
                elif point[0] == x_val:
                    self.right_full.append(point)
                    self.left_full.append(point)

        self.left_full += c_b
        self.right_full += c_b
        # the following may have to be inverted in order to work
        """if len(self.inner) < 5:
            for point_collection in self.inner:  # may have more than one inner contour. But if len(self.inner) < 5 know that we need to treat it as a case of multiple inner loops
                for point in point_collection:
                    if point[0] < x_val:  # gives IndexError (invalid index to scalar variable)
                        self.left_full.append(point)
                    elif point[0] > x_val:
                        self.right_full.append(point)
                    elif point[0] == x_val:
                        self.right_full.append(point)
                        self.left_full.append(point)
        else:
            for point in self.inner:
                if point[0] < x_val:
                    self.left_full.append(point)
                elif point[0] > x_val:
                    self.right_full.append(point)
                elif point[0] == x_val:
                    self.right_full.append(point)
                    self.left_full.append(point)"""

    def getLeftFull(self):
        return self.left_full

    def getRightFull(self):
        return self.right_full

    def setUpperLeft(self, points):
        self.upper_left = points

    def setLowerLeft(self, points):
        self.lower_left = points

    def setUpperRight(self, points):
        self.upper_right = points

    def setLowerRight(self, points):
        self.lower_right = points

    def setLeft(self, c_b):
        self.left = self.upper_left + self.lower_left + c_b

    def setRight(self, c_b):
        self.right = self.upper_right + self.lower_right + c_b

    def setInner(self, inner):
        self.inner.append(inner)

    def getInner(self):
        return self.inner

    def getCategory(self):
        return self.category

    def setCuttingPoints(self, cutting_lower, cutting_upper):
        self.cutting_points_upper = cutting_upper
        self.cutting_points_lower = cutting_lower

    def getCuttingPoints(self):
        return self.cutting_points_lower, self.cutting_points_upper

    def setCategory(self, category):
        self.category = category

    def getWidth(self):
        print str(self.outer[2]-self.outer[0])

        return int(self.outer[2] - self.outer[0])

    def getXmin(self):
        return self.outer[0]

    def getOuter(self):
        return self.outer[1]

    def getXmax(self):
        return self.outer[2]

    def setUpper(self, upper):
        self.upper = upper

    def setLower(self, lower):
        self.lower = lower

    def getUpper(self):
        return self.upper

    def getLower(self):
        return self.lower

    def printContourInfo(self):
        print "This outer contour has " + str(len(self.inner)) + " inner contours belonging to it."

    def printWidth(self):
        print str(self.outer[max(0, len(self.outer)-1)][0] - self.outer[0][0])

    def drawStuff(self, black_rectangle):  # may not need to use the drawContours function explicitly
        black_rectangle = black_rectangle
        contour = self.outer[1]
        # black_rectangle[:, :, 0] = 255  # R, G, B in this order for 0, 1, 2.
        for point in contour:
            # for point in point_collection:
                # black_rectangle[point[1]:point[1], point[0]:point[0], 2] = 255  # not being displayed properly
            black_rectangle[point[1], point[0], 2] = 255  # not being displayed properly
                # contour points not appropriately drawn.
        for point_collection in self.inner:
            for point in point_collection[1]:
                black_rectangle[point[1], point[0], 0] = 255
                # indication that drawContours() does more than simply draw the particular points in the colelction?
            # black_rectangle[point[0], point[1], 2] = 255
        # cv2.drawContours(black_rectangle, self.outer, 1, color=(0, 255, 0), thickness=1)
        # cv2.drawContours(black_rectangle, self.inner, -1, color=(0, 0, 255), thickness=1)
        return black_rectangle

    def drawUpper(self, black_rectangle):
        black_rectangle = black_rectangle
        upper = self.getUpper()

        for point in upper:
            black_rectangle[point[1], point[0], 1] = 255

        return black_rectangle

    def drawLower(self, black_rectangle):
        black_rectangle = black_rectangle
        lower = self.getLower()

        for point in lower:
            black_rectangle[point[1], point[0], 0] = 255
        return black_rectangle

    def drawCuts(self, black_rectangle):
        black_rectangle = black_rectangle
        lower_cut, upper_cut = self.getCuttingPoints()

        # have 4 kinds of cases:
        # delta x = 0 (straight vertical line)
        # abs(delta_x) < abs(delta_y)
        # abs(delta_y) < abs(delta_x)
        # abs(delta_x) == abs(delta_y)

        # compute delta x. Practically speaking, when would it be different from 0?
        for point_l in enumerate(lower_cut):  # enumerate is used if want to get the indeces. On form [index, (point)]
            if point_l[1][0] == upper_cut[point_l[0]][0]:  # Case 1 i.e. they are on the same x-coordinate
                black_rectangle[upper_cut[point_l[0]][1]:point_l[1][1], point_l[1][0], 2] = 255
            else:
                print "We actually have a case where delta_x is different from 0! Time to implement this part in drawCuts()"
            # need to handle cases where the pts are not vertically aligned
        return black_rectangle

    def getCuttingLine(self, lower_cut, upper_cut):  # would like the cuts to correspond to one x-value only
        # lower_cut, upper_cut = self.getCuttingPoints()
        # at this point we have only the end points of the lines

        # want to return all the points along the lines
        c_b = []
        if lower_cut[0] == upper_cut[0]:  # for other scenarios, skip for now.
            for y in xrange(min(upper_cut[1], lower_cut[1]), max(lower_cut[1], upper_cut[1])+1):
                c_b.append([lower_cut[0], y])
                # in worst case scenario, return one single point
        return c_b  # in very rare cases will return a single element

    def drawLeft(self, black_rectangle):
        for point in self.left:
            black_rectangle[point[1], point[0], 1] = 255

        return black_rectangle

    def drawRight(self, black_rectangle):
        for point in self.right:
            black_rectangle[point[1], point[0], 1] = 255

        return black_rectangle

    def drawLeftRightFull(self, black_rectangle):
        left = self.left_full
        right = self.right_full

        for point in left:
            #print point
            black_rectangle[point[1], point[0], 0] = 255

        for point in right:
            #print point
            black_rectangle[point[1], point[0], 2] = 255

        return black_rectangle

    def getXvals(self):
        outer = self.outer
        return outer[:, 0]

    def getYvals(self):
        outer = self.outer
        return outer[:, 1]

def Fujisawa(filename_slice):
    img = cv2.imread(filename_slice, 0)
    binary = 255 - img
    black_bg = np.zeros(binary.shape, dtype='uint8')

    im2, cnt, hier = cv2.findContours(binary.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    large_cnt = []

    for c in cnt:
        if (cv2.contourArea(c) > 120) and (cv2.contourArea(c)<0.95*img.shape[0]*img.shape[1]):  # may use a smaller treshold since we do not have a large image, need not consider PT
            large_cnt.append(c)  # or consider a threshold that varies with the height of the entire extracted image

    large_cnt_backup = large_cnt
    cv2.drawContours(black_bg, large_cnt, -1, color=255, thickness=1)
    # Image.fromarray(black_bg).show()

    # next sort the contours such that points are from left to right
    # first, need to reshape the list of contours, as each coordinate currently is on the form [[x, y]] instead of [x, y]
    # reshape the list of contours

    arr = []
    large_cnt = np.asarray(large_cnt)

    for element in large_cnt:
        points = []
        for point in element:
            points.append(point[0])
        arr.append(np.asarray(points))
    arr = np.asarray(arr)

    arr_anti_clockwise = []
    large_cnt_backup = np.asarray(large_cnt_backup)

    for element in large_cnt_backup:  # keep track of the points in anti-clockwise direction (paper assumes anti-clockwise)
        points = []
        for point in element:
            points.append(point[0])
        arr_anti_clockwise.append(np.asarray(points))
    arr_anti_clockwise = np.asarray(arr_anti_clockwise)

    # desired shape obtained
    # now sort each element
    sorted_contours = []
    for contour in arr:
        sorted_contours.append(sorted(contour, key=lambda x: x[0]))  # sorting useful for finding min, max but not for much else
        # sorted_contours.append(np.sort(contour, axis=0))  # appears to sort both x and y coordinates simultaneously
    # use a different sorting function, with a lambda instead

    # correctly sorted by this point, according to x-coordinate
    # next sort all the contours from left to right - according to their minimum x-value

    x_min = []
    x_max = []
    for contour in sorted_contours:
        x_min.append(contour[0][0])
        x_max.append(contour[len(contour)-1][0])
    # print x_min
    # print x_max
    # no need to store the value of x_min for long term, since contour points are all sorted in order of increasing x
    # zipped = zip(x_min, x_max, sorted_contours)
    zipped = zip(x_min, arr_anti_clockwise, x_max)
    sorted_zip = sorted(zipped, key=lambda x: x[0])  # sorted_zip now contains a list of contours sorted from left to right
    # next, want to determine whether the contours are inner or outer contours
    # let 0 denote outer contours, and 1 denote inner contours. Want to create a list P
    # to check if the CC is inner or outer,
    # check x_min, x_max, y_min, y_max. However, taking y-values into account might be problematic, ignore for now
    # Note: One outer contour may have more than one inner contour, e.g. in the number 8

    # first pass: find out which contours are outer and which are inner
    outer_inner = []
    for k in xrange(1, len(sorted_zip)):  # the checking conditions need to be updated to involve x_max now
        if sorted_zip[k][2] < sorted_zip[k-1][2]:  # in this case it's INNER
            outer_inner.append(1)
        else:
            outer_inner.append(0)
    # skips the first contour
    # print outer_inner
    # next, combine the outer_inner information with the contour information. No longer need x_min
    # but may extract in case it's needed
    x_min, contours, x_max = zip(*sorted_zip)  # x_min will be ditched
    outer_inner.insert(0, 0)  # assumes the first contour will always be an outer contour, which is reasonable
    sorted_outer_inner_zip = zip(x_min, contours, x_max, outer_inner)

    # second pass: want to find the parent contour of each inner contour

    pattern_components = []

    outer_contours = []
    inner_contours = []
    count_outer = -1
    for contour in sorted_outer_inner_zip:  # may not be necessary to pass on the outer_inner value, but it does not hurt at this point
        if contour[3] == 0:  # i.e. if it's outer
            pattern_components.append(PatternComponent(contour, img.shape[1]))  # create a new instance of pattern components, and keep in array
            count_outer += 1
        else:  # assumes the the first contour to be checked will always be an outer contour
            pattern_components[count_outer].setInner(contour)
            # inner_contours.append(contour)  # any inner should belong to the previous outer?
    for pc in pattern_components:
        pc.printContourInfo()

    # next, want to draw ihe contours in different colors for the sake of visibility
    black_rectangle = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    for pc in pattern_components:
        pc.drawStuff(black_rectangle)  # draws who knows what

    # Image.fromarray(black_rectangle).show()
    # successfully separated outer and inner contours

    # check if is a touching pattern or not

    # set thresholds based on the normal width of a character. Need to calibrate based on the actual data
    t_w_1 = 250  # if > than this, will say it IS touching
    t_w_2 = 80  # if t_w_1 > w > t_w_2, it may or may not be touching

    # hyphens: Not considered by Fujisawa et al, need to handle them separately
    # hyphens would not actually be considered as touching unless they are in fact touching
    # collect the values of the widths of chars:
    for pc in pattern_components:
        if pc.getWidth() > t_w_1:  # the Width function needs to be updated now that the points are clockwise
            pc.setCategory(2)
        elif pc.getWidth() < t_w_2:
            pc.setCategory(0)
        else:
            pc.setCategory(1)

    # separate each contour into upper and lower, based on the leftmost and rightmost points

    # verify that the contours are indeed drawn in correct order, i.e. connecting points only and no jumping
    """check_direction = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    # want to gradually increase the intensity for each point
    min_intensity = 30

    for pc in pattern_components:
        k = 0
        outer = pc.getOuter()
        for point in outer:
            check_direction[point[1], point[0], 0] = min(min_intensity + k, 200)
            k += 1

    Image.fromarray(check_direction).show()"""

    # want to split into lower and upper. the starting point of the contour points is by the top, however, not at the leftmost point
    # several ways to approach this, could "rotate" the pts

    # Find the indices of the leftmost and rightmost points. Need not be accurate, just do the first point that matches x_min or x_max
    for pc in pattern_components:
        # print pc.getOuter()  # gets first element only, while what we actually want is to get all the x-values.
        outer = pc.getOuter()
        index_x_min = [row[0] for row in outer].index(pc.getXmin())
        index_x_max = [row[0] for row in outer].index(pc.getXmax())
        # print index_x_max - index_x_min  # all values are > 0

        pc.setUpper(np.asarray(list(reversed(outer[index_x_max:len(outer)])) + list(reversed(outer[0:index_x_min]))))
        pc.setLower(np.asarray(outer[index_x_min:index_x_max]))  # is not reversed at all, as the order is correct by default

        # may want to check if any coordinates are overlapping between both the upper and lower lists?
        # confirmed by the below: No points are overlapping
        """upper = pc.getUpper()
        lower = pc.getLower()
        for u in upper:
            for l in lower:
                if (u[0] == l[0]) and (u[1] == l[1]):
                    print u
                    print l"""

        # need a better comparison method - the two sets of points are on a different form
        # upper may be list, while lower is array?

    # want to find index matching pre-determined values for min and max (stored in the pattern component object)
    # for each lower and each upper, want to reduce the number of points for each x-coordinate to 1
    # lower: choose uppermost. upper: choose the lowest

    for pc in pattern_components:
        lower = pc.getLower()  # want to ultimately use pc.setLower() with the new collection
        upper = pc.getUpper()
        # the current order of points is such that those w/ same x-coord may NOT be directly after each other
        # could create a copy, sort by x-coord, and pop some elements from the original list?
        lower_increasing_x = np.asarray(sorted(lower, key=lambda x: x[0]))
        upper_increasing_x = np.asarray(sorted(upper, key=lambda x: x[0]))

        #print lower_increasing_x
        #print upper_increasing_x
        # for each x-value, extract a collection of all the points
        keep_points = []
        x_values = []
        for element in lower_increasing_x:  # first pass: find all the x-values
            if element[0] not in x_values:
                x_values.append(element[0])
        #print x_values

        # next, handle the upper contour points, want to find the lowest y-coordinate for each x-coordinate (lowest meaning largest y-value)
        for x in x_values:  # may be able to merge some for-loops in the above to cut down PT by a nano-second
            current = []
            for point in lower:
                if point[0] == x:
                    current.append(point)

            y_vals_current = []
            for point in current:
                y_vals_current.append(point[1])

            index_min_y = np.argmin(y_vals_current, axis=0)  # check if works correctly?

            keep_points.append(current[index_min_y])
        #print keep_points
        pc.setLower(np.asarray(keep_points))

        keep_points = []
        x_values = []
        for element in upper_increasing_x:
            if element[0] not in x_values:
                x_values.append(element[0])
        #print x_values

        for x in x_values:
            current = []
            y_vals_current = []
            for point in upper:
                if point[0] == x:
                    current.append(point)
                    y_vals_current.append(point[1])

            # print len(y_vals_current)  # getting into trouble once there are no y-values corresponding to the y_value
            # for whatever reason, for some value(s) x there is not any y-value.
            index_max_y = np.argmax(y_vals_current, axis=0)  # current error: empty list

            keep_points.append(current[index_max_y])
        #print keep_points
        pc.setUpper(np.asarray(keep_points))

    black_rectangle = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

    for pc in pattern_components:
        black_rectangle = pc.drawUpper(black_rectangle)

    # Image.fromarray(black_rectangle).show()

    black_rectangle = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

    for pc in pattern_components:
        black_rectangle = pc.drawLower(black_rectangle)

    # Image.fromarray(black_rectangle).show()

    black_rectangle = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

    for pc in pattern_components:
        black_rectangle = pc.drawLower(black_rectangle)
        black_rectangle = pc.drawUpper(black_rectangle)

    # Image.fromarray(black_rectangle).show()
    h_t = 75  # assumes a constant value for h_t. May be better to choose a dynamic threshold h_t

    knn_model = loadKNN()  # successfully loads

    for pc in pattern_components:
        if pc.getCategory() > 0:  # only consider the potentially touching characters
            # next: the hypothesis that no cuts are appropriate
            outer = pc.getOuter()  # is a collection of point (desired form)
            outer_pts = []
            for point in outer:
                x = point[0]
                y = point[1]
                outer_pts.append([x, y])
            outer = outer_pts
            inner = pc.getInnerContours()  # is a collection of arrays (not desired)
            # inner is a list of arrays, want a list of pts
            inner_pts = []
            for element in inner:
                x = element[0]
                y = element[1]

                inner_pts.append([x, y])

            inner = inner_pts
            #print "Inner: "
            #print inner

            if len(inner) > 0:
                all_pts = outer + inner  # at this point, inner should be just a list of points, but it does not appear to hold for the 00 contour.
            else:
                all_pts = outer
            x_min, x_max, y_min, y_max = pc.findBoundingBox(all_pts)
            new_size = max((x_max - x_min), (y_max - y_min))
            # may be able to recycle some code here

            segment_c = binary[y_min:y_max, x_min:x_max]
            black_bg = np.zeros((new_size, new_size), dtype='uint8')

            if segment_c.shape[0] > segment_c.shape[1]:  # if it's taller than it's wide
                x_min = int(new_size / 2) - int(1.0 * segment_c.shape[1] / 2)
                x_max = x_min + segment_c.shape[1]
                black_bg[0:black_bg.shape[0], x_min:x_max] = segment_c
                char_on_bg_c = black_bg
            elif segment_c.shape[0] == segment_c.shape[1]:  # just place directly on top.
                char_on_bg_c = segment_c
            else:  # if wider than it's tall
                y_min = int(new_size / 2) - int(1.0 * segment_c.shape[0] / 2)
                y_max = y_min + segment_c.shape[0]
                black_bg[y_min:y_max, 0:black_bg.shape[0]] = segment_c
                char_on_bg_c = black_bg

            dim_temp = 128
            resized_c = cv2.resize(char_on_bg_c, (dim_temp, dim_temp), cv2.INTER_LINEAR)
            char_dim = 28
            deskew_c = deskew(resized_c)
            trim_c = trim_padding(deskew_c)
            resized_c = resize_with_constant_ratio(trim_c, char_dim=char_dim)
            pad_c = pad_digit(resized_c, char_dim=char_dim)
            # Image.fromarray(pad_c).show()
            # prepare pad_c for KNN
            arr_c = np.asarray(pad_c)
            ready_c = arr_c.reshape(-1, 784).astype('float32')
            ret, result_c, neighbours, dist = knn_model.findNearest(ready_c, k=3)  # k may need to be changed
            print "Result without cuts: "
            print result_c
            print dist
            pc.setSymbolGuesses(str(int(result_c)))
            pc.setSymbolDistances(dist)
            # should not repeat the above every single time there is a new cut. Rather, do once for each pc
            # for the potentially touching, want to obtain H(x)
            upper = pc.getUpper()
            lower = pc.getLower()

            # upper_x = upper[:, 0]  # cannot be used to access elements
            # lower_x = lower[:, 0]

            outer = pc.getOuter()  # for getting all the x-values
            h_x = []
            # collect all the x-values
            x_vals = []
            for point in outer:
                if point[0] not in x_vals:
                    x_vals.append(point[0])
            # have all x-vals now, next find the value of H(x) for each x
            # print x_vals  # x-vals are not in a particular order
            for x in x_vals:
                try:
                    y_u = upper[[row[0] for row in upper].index(x)][1]
                except ValueError:
                    y_u = 0

                try:
                    y_l = lower[[row[0] for row in lower].index(x)][1]
                except ValueError:
                    y_l = 0
                # index_u = [row[0] for row in upper].index(x)
                # index_l = [row[0] for row in lower].index(x)

                h = abs(y_u - y_l)
                h_x.append((x, h))
            # print h_x  # currently works as intended
            x_for_graph = [row[0] for row in h_x]
            y_for_graph = [row[1] for row in h_x]

            """fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            plt.plot(x_for_graph, y_for_graph, "ro")  # plots red dots for H(x)
            plt.show()"""

            # look for the points where H(x) and h_t cross
            h_x_x_min = pc.getXmin()
            h_x_x_max = pc.getXmax()
            # collect max and min x to establish boundaries
            r_bar = h_x_x_max - h_x_x_min
            x_bar = int((h_x_x_min + h_x_x_max)/2.0)

            lower_bound = int(x_bar - (1.0*r_bar/3))  # may want to change from 3 to larger value
            upper_bound = int(x_bar + (1.0*r_bar/3))

            #print "Lower bound: " + str(lower_bound)
            #print "Upper bound: " + str(upper_bound)

            cross_points_upper = []
            cross_points_lower = []
            for x in xrange(lower_bound, upper_bound):  # except that H(x) is defined for all values of x
                if (h_x[[row[0] for row in h_x].index(x)][1] >= h_t) and (h_x[[row[0] for row in h_x].index(x)-1][1] < h_t): # a golden cross
                    cross_points_lower.append(lower[[row[0] for row in lower].index(x)])
                    cross_points_upper.append(upper[[row[0] for row in upper].index(x)])
                elif (h_x[[row[0] for row in h_x].index(x)][1] <= h_t) and (h_x[[row[0] for row in h_x].index(x)-1][1] > h_t):  # cross from over to under
                    cross_points_lower.append(lower[[row[0] for row in lower].index(x)])
                    cross_points_upper.append(upper[[row[0] for row in upper].index(x)])

            # print cross_points_lower  # keeps track of all the x-values
            # print cross_points_upper

            black_rectangle = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
            black_rectangle = pc.drawUpper(black_rectangle)
            black_rectangle = pc.drawLower(black_rectangle)

            for point in cross_points_lower:
                black_rectangle[point[1], point[0], :] = 125

            for point in cross_points_upper:
                black_rectangle[point[1], point[0], :] = 67

            # Image.fromarray(black_rectangle).show()
            cutting_points_lower = []
            cutting_points_upper = []

            # want to shift the cutting points to a value x such that the vertical width expands abruptly
            # for now, use H(x), as it seems reliable enough (would only case troubles where the upper contour point is below the lower contour point)
            # how large region to search??

            delta_thresh = 1  # for 100% change to terminate immediately
            # may want one while loop for each candidate point, i.e. while within for
            for point in cross_points_lower:
                x_candidate = point[0]
                x_cur_highest_delta = x_candidate
                cur_highest_delta = 0
                delta_x = 1
                searching = True
                interval_length = int(r_bar / 5)  # tried r_bar over 10, which appears to be too narrow

                while searching:
                    h_x_cur = h_x[[row[0] for row in h_x].index(x_candidate)][1]
                    h_x_delta_neg = h_x[[row[0] for row in h_x].index(x_candidate - delta_x)][1]
                    h_x_delta_pos = h_x[[row[0] for row in h_x].index(x_candidate + delta_x)][1]
                    # can either use delta_x or x_cur for the increment. Here use delta_x, so need to be careful when extract the cutting pts
                    ratio_pos = abs((1.0*h_x_delta_neg - h_x_cur)/h_x_cur)
                    ratio_neg = abs((1.0*h_x_delta_pos - h_x_cur)/h_x_cur)

                    if ratio_pos > delta_thresh:
                        searching = False
                        x_cur_highest_delta = x_candidate + delta_x - 1
                        # cutting_points_lower.append(lower[[row[0] for row in lower].index(h_x_cur+delta_x-1)])  # need to add the x,y of the cutting point from lower (have done pc.getLower()
                        # cutting_points_upper.append(upper[[row[0] for row in upper].index(h_x_cur+delta_x-1)])  # similar, just for upper instead

                    if ratio_neg > delta_thresh:
                        searching = False
                        x_cur_highest_delta = x_candidate - delta_x + 1
                        # cutting_points_lower.append(lower[[row[0] for row in lower].index(
                            # h_x_cur - delta_x + 1)])  # need to add the x,y of the cutting point from lower (have done pc.getLower()
                        # cutting_points_upper.append(
                            # upper[[row[0] for row in upper].index(h_x_cur - delta_x + 1)])  # similar, just for up

                    if ratio_pos > cur_highest_delta:
                        cur_highest_delta = ratio_pos
                        x_cur_highest_delta = x_candidate + delta_x - 1

                    if ratio_neg > cur_highest_delta:
                        cur_highest_delta = ratio_neg
                        x_cur_highest_delta = x_candidate - delta_x + 1
                    # for now, we are ignoring cases where more than one point is having equally large delta (is same as taking the first x-value in the searching region that satisfies criteria)

                    delta_x += 1

                    if delta_x > interval_length:
                        searching = False

                cutting_points_lower.append(lower[[row[0] for row in lower].index(x_cur_highest_delta)])
                cutting_points_upper.append(upper[[row[0] for row in upper].index(x_cur_highest_delta)])

            for point in cutting_points_lower:
                black_rectangle[point[1], point[0], 2] = 255

            for point in cutting_points_upper:
                black_rectangle[point[1], point[0], 2] = 255

            pc.setCuttingPoints(cutting_points_lower, cutting_points_upper)
            black_rectangle = pc.drawCuts(black_rectangle)

            # Image.fromarray(black_rectangle).show()

            if len(pc.getCuttingPoints()) == 0 and len(pc.getCuttingPoints()) == 0:
                # check if there is moe than one inner loop.
                inner_cnts = pc.getInner()
                if len(inner_cnts) > 1: # have mroe than one inner loop
                    print "Want to treat this with algorithm part II "                # go to algorithm part 2
                elif h_x[[row[0] for row in h_x].index(x_bar)] < h_t: # for handling cases where in the entire interval the value of H(x) is below h_t, set cutting point to be the midpoint of the interval
                    cutting_points_lower.append(lower[[row[0] for row in lower].index(x_bar)])
                    cutting_points_upper.append(upper[[row[0] for row in upper].index(x_bar)])
                    print "Set cut to be midpoint of the interval X_1 to X_2"
                else:
                    print "Did none of the above, still have no candidate cutting points! "

            # next, may want to look into the multiple hypothesis
            # will first do the cutting: separate the points based on the previous steps, and use recognition-based to tests to check what is the most likely correct value
            for point_l in cutting_points_lower:  # assumes the x-values are the same
                x_val = point_l[0]
                c_b = pc.getCuttingLine(point_l, cutting_points_upper[[row[0] for row in cutting_points_upper].index(point_l[0])])
                # the two lines below may not be necessary
                lower = pc.getLower()
                upper = pc.getUpper()

                # use a simple for loop based on x values to split into left and righ
                lower_left = []
                lower_right = []
                upper_left = []
                upper_right = []

                for point in lower:
                    if point[0] < x_val: # skip equality since it's a part of x_b anyways
                        lower_left.append(point)
                    elif point[0] > x_val:
                        lower_right.append(point)

                for point in upper:
                    if point[0] < x_val:
                        upper_left.append(point)
                    elif point[0] > x_val:
                        upper_right.append(point)

                pc.setUpperLeft(upper_left)
                pc.setUpperRight(upper_right)
                pc.setLowerLeft(lower_left)
                pc.setLowerRight(lower_right)

                pc.setLeft(c_b)
                pc.setRight(c_b)

                pc.setLeftRightFull(c_b)

                black_bg = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
                # Image.fromarray(pc.drawLeftRightFull(black_bg)).show()

                # when draw, should take pts from the entire character, not just the upper and lower
                # also, have some weird cases of green horizontal lines that keep being printed/Drawn

                black_bg = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
                # Image.fromarray(pc.drawLeft(black_bg)).show()
                # Image.fromarray(pc.drawRight(black_bg)).show()

                # now want to extract the hypothesised characters from each pattern component
                left_full = pc.getLeftFull()
                x_min, x_max, y_min, y_max = pc.findBoundingBox(left_full)
                segment_l = binary[y_min:y_max, x_min:x_max]  # is black on white
                # segment_l = 255 - segment_l  # should now be white on black
                # Image.fromarray(segment_l).show()

                right_full = pc.getRightFull()
                x_min, x_max, y_min, y_max = pc.findBoundingBox(right_full)
                segment_r = binary[y_min:y_max, x_min:x_max]
                # segment_r = 255 - segment_r
                # Image.fromarray(segment_r).show()

                # note: also need to test the possibility that no cuts are appropriate for this pattern component
                # start extracting the chars: LHS
                new_size = max(segment_l.shape[0], segment_l.shape[1])  # should still work even after inverting colors of binary segment
                black_bg = np.zeros((new_size, new_size), dtype='uint8')  # black bg

                if segment_l.shape[0] > segment_l.shape[1]:  # if it's taller than it's wide
                    x_min = int(new_size/2) - int(1.0*segment_l.shape[1]/2)
                    x_max = x_min + segment_l.shape[1]
                    black_bg[0:black_bg.shape[0], x_min:x_max] = segment_l
                    char_on_bg_l = black_bg
                elif segment_l.shape[0] == segment_l.shape[1]:  # just place directly on top.
                    char_on_bg_l = segment_l
                else:  # if wider than it's tall
                    y_min = int(new_size/2) - int(1.0*segment_l.shape[0]/2)
                    y_max = y_min + segment_l.shape[0]
                    black_bg[y_min:y_max, 0:black_bg.shape[0]] = segment_l
                    char_on_bg_l = black_bg

                # Now do RHS:
                new_size = max(segment_r.shape[0], segment_r.shape[
                    1])  # should still work even after inverting colors of binary segment
                black_bg = np.zeros((new_size, new_size), dtype='uint8')  # black bg

                if segment_r.shape[0] > segment_r.shape[1]:  # if it's taller than it's wide
                    x_min = int(new_size / 2) - int(1.0 * segment_r.shape[1] / 2)
                    x_max = x_min + segment_r.shape[1]
                    black_bg[0:black_bg.shape[0], x_min:x_max] = segment_r
                    char_on_bg_r = black_bg
                elif segment_r.shape[0] == segment_r.shape[1]:  # just place directly on top.
                    char_on_bg_r = segment_r
                else:  # if wider than it's tall
                    y_min = int(new_size / 2) - int(1.0 * segment_r.shape[0] / 2)
                    y_max = y_min + segment_r.shape[0]
                    black_bg[y_min:y_max, 0:black_bg.shape[0]] = segment_r
                    char_on_bg_r = black_bg

                # Have both left and right sides at this point. Next, do the resizing, deskewing etc
                # resize
                # use a size 128 for the purpose of smoothening the deskewing process
                size_temp = 128
                resized_l = cv2.resize(char_on_bg_l, (size_temp, size_temp), cv2.INTER_LINEAR)
                resized_r = cv2.resize(char_on_bg_r, (size_temp, size_temp), cv2.INTER_LINEAR)

                # Image.fromarray(resized_l).show()
                # Image.fromarray(resized_r).show()

                # print "Have successfully resized stuff! "
                # they are now 20, 20 pixels

                char_dim = 28  # for the total box with the 4-wide frame
                # next step is to deskew
                deskew_l = deskew(resized_l)
                deskew_r = deskew(resized_r)

                trim_pad_l = trim_padding(deskew_l)
                trim_pad_r = trim_padding(deskew_r)

                #resized_l = resize_with_constant_ratio(trim_pad_l, char_dim=char_dim)  # is this even necessary?
                # resized_r = resize_with_constant_ratio(trim_pad_r, char_dim=char_dim)

                resized_l = cv2.resize(trim_pad_l, (char_dim, char_dim), cv2.INTER_LINEAR)
                resized_r = cv2.resize(trim_pad_r, (char_dim, char_dim), cv2.INTER_LINEAR)

                padded_l = pad_digit(resized_l, char_dim=char_dim)  # needed or not?
                padded_r = pad_digit(resized_r, char_dim=char_dim)  # expect it to be needed only if the digit is not squared?

                # Image.fromarray(255 - padded_l).show()
                # Image.fromarray(255 - padded_r).show()

                # Image.fromarray(resized_l).show()
                # Image.fromarray(resized_r).show()

                # do knn classification of each image
                # need to add way of keeping track of the distances, though

                #print resized_l.shape
                #print resized_r.shape

                arr_l = np.asarray(padded_l)
                ready_l = arr_l.reshape(-1, 784).astype('float32')
                ret, result_l, neighbours, dist_l = knn_model.findNearest(ready_l, k=3)  # k may need to be changed
                print "KNN result: "
                print result_l
                print dist_l

                arr_r = np.asarray(padded_r)
                ready_r = arr_r.reshape(-1, 784).astype('float32')
                ret, result_r, neighbours, dist_r = knn_model.findNearest(ready_r, k=3)  # k may need to be changed
                print "KNN result: "
                print result_r
                print dist_r

                pc.setSymbolGuesses(str(int(result_l)) + str(int(result_r)))
                # setting the distances is not straightforward when have two lists of distances
                # pc.setSymbolDistances([min(dist_l) + min(dist_r)])  # is not int at this point
                # skip the conversion to integer
                pc.setSymbolDistances([(min(dist_l) + min(dist_r))/2])

        elif pc.getCategory() == 0:  # check if it's a hyphen
            x_min, x_max, y_min, y_max = pc.findBoundingBox(pc.getOuter())
            if (cv2.contourArea(pc.getOuter()) > (0.70 * ((x_max-x_min)*(y_max-y_min)))) and (x_max-x_min > y_max-y_min):  # then go ahead and assume it's a hyphen
                pc.setSymbolGuesses("-")  # add hyphen as guess
                pc.setSymbolDistances([0, 1])  # will therefore always be selected
            else:  # treat as if no cuts are necessary (no cut hypothesis). Rarely triggered
                outer = pc.getOuter()
                inner = pc.getInnerContours()  # may have multiple inner loops - in this case want to combine all of them
                # handle the different formats of outer, inner
                outer_pts = []
                for point in outer:
                    x = point[0]
                    y = point[1]
                    outer_pts.append([x, y])
                outer = outer_pts

                inner_pts = []
                for point in inner:
                    x = point[0]
                    y = point[1]
                    inner_pts.append([x, y])
                inner = inner_pts
                # the parts here may also be bugged, see line # 995

                all_pts = outer + inner
                x_min, x_max, y_min, y_max = pc.findBoundingBox(all_pts)
                new_size = max((x_max-x_min), (y_max-y_min))
                # may be able to recycle some code here
                segment_c = binary[y_min:y_max, x_min:x_max]
                black_bg = np.zeros((new_size, new_size), dtype='uint8')

                if segment_c.shape[0] > segment_c.shape[1]:  # if it's taller than it's wide
                    x_min = int(new_size/2) - int(1.0*segment_c.shape[1]/2)
                    x_max = x_min + segment_c.shape[1]
                    black_bg[0:black_bg.shape[0], x_min:x_max] = segment_c
                    char_on_bg_c = black_bg
                elif segment_c.shape[0] == segment_c.shape[1]:  # just place directly on top.
                    char_on_bg_c = segment_c
                else:  # if wider than it's tall
                    y_min = int(new_size/2) - int(1.0*segment_c.shape[0]/2)
                    y_max = y_min + segment_c.shape[0]
                    black_bg[y_min:y_max, 0:black_bg.shape[0]] = segment_c
                    char_on_bg_c = black_bg

                dim_temp = 128
                resized_c = cv2.resize(char_on_bg_c, (dim_temp, dim_temp), cv2.INTER_LINEAR)
                char_dim = 28
                deskew_c = deskew(resized_c)
                trim_c = trim_padding(deskew_c)
                resized_c = resize_with_constant_ratio(trim_c, char_dim=char_dim)
                pad_c = pad_digit(resized_c, char_dim=char_dim)

                # Image.fromarray(pad_c).show()
                # prepare pad_c for KNN
                arr_c = np.asarray(pad_c)
                ready_c = arr_c.reshape(-1, 784).astype('float32')
                ret, result_c, neighbours, dist = knn_model.findNearest(ready_c, k=3)  # k may need to be changed
                print "Result without cuts: "
                print result_c
                print dist
                pc.setSymbolGuesses(str(int(result_c)))
                pc.setSymbolDistances(dist)
                # at this point, should be ready to extract characters (segmented)
                # caution: need to add the hypothesis that the character is not segmented at all

    for pc in pattern_components:
        # print pc.getCategory()
        print pc.getSymbolGuesses()  # is this list ordered?

    stri = ""
    for pc in pattern_components:  # want to print the best guess for the value of the line before comparing with ASN
        stri += str(pc.getMinSymbol())

    print "The best guess is: "
    print stri

    for pc in pattern_components:
        print pc.getBeta()

    # next, use beta values to form the guesses for item code value
    stri_0 = ""
    stri_1 = ""
    stri_2 = ""

    for pc in pattern_components:
        if pc.getBeta() == 0:
            stri_0 += pc.getMinSymbol()
            stri_1 += pc.getMinSymbol()
            stri_2 += pc.getMinSymbol()
        elif pc.getBeta() == 1:
            stri_1 += pc.getMinSymbol()
            stri_2 += pc.getMinSymbol()
        elif pc.getBeta() == 2:
            stri_2 += pc.getMinSymbol()

    print stri_0
    print stri_1
    print stri_2

    text_cands = []
    text_cands.append(stri_0)
    text_cands.append(stri_1)
    text_cands.append(stri_2)
    print "Ranked symbol guesses: "
    for pc in pattern_components:
        print pc.getRankedSymbolGuesses()  # seems to have no effect?
    # now, generate all the text outputs

    return text_cands
