import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from skimage import filters
from tkFileDialog import askopenfilename
from scipy import ndimage as ndi
import math
from tesserocr import PyTessBaseAPI, iterate_level, RIL
from skimage import feature
import matplotlib.image as mpimg


filename = askopenfilename()
print filename  # for debugging purposes
img = cv2.imread(filename)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

contour_area_min = 350
contour_area_max = int(math.floor(img.shape[0]*img.shape[1]*0.08))
strel_size = 9  # may have some correlation with the min contour area threshold.

img_red = img[:, :, 0]
img_green = img[:, :, 1]
img_blue = img[:, :, 2]

edge_red = feature.canny(img_red, sigma=2)  # alternatively, add lower and upper thresholds of 50/100 TBC
edge_green = feature.canny(img_green, sigma=3)  # changed from 3
edge_blue = feature.canny(img_blue, sigma=3)  #may consider using default sigma value instead

#edge_red = cv2.Canny(img_red, 50, 100)
#edge_green = cv2.Canny(img_green, 50, 100)
# edge_blue = cv2.Canny(img_blue, 50, 100)

edge_assimilated = np.logical_or(edge_red, np.logical_or(edge_green, edge_blue))  # boolean array

strel1 = np.zeros((strel_size, strel_size))
for i in range(0, strel_size):
    strel1[int(math.floor(strel_size/2.0))][i] = 1
# horizontal dilation
edge_dim_1 = ndi.binary_dilation(edge_assimilated, structure=strel1)

strel2 = np.zeros((strel_size, strel_size))

for j in range(0, strel_size):
    strel2[j][int(math.floor(strel_size/2.0))] = 1
# vertical dilation
edge_dim_2 = ndi.binary_dilation(edge_assimilated, structure=strel2)
# combine the results with OR operator
edges_dil_comb = np.logical_or(edge_dim_1, edge_dim_2)
# all_labels = measure.label(edges_dil_comb, neighbors=8, connectivity=2).astype('uint8')  # same dimensions as image
all_labels = edges_dil_comb.astype('uint8')
# the above step may not be necessary? If skip, convert edges_dil_comb to 'uint8'
im2, abac, hierarchy = cv2.findContours(all_labels.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

large_contours = []
for contour in abac:
    if cv2.contourArea(contour) > contour_area_min and (cv2.contourArea(contour) < contour_area_max):  # Green's theorem for contour area approximation
        large_contours.append(contour)
# at this point have, collected large contours only.
# instead of drawing contours, try to select the interior pts of each contour, in grayscale

collect_gs = 255 + np.zeros((img.shape[0], img.shape[1]), dtype='uint8')  # want grayscale
black_shell = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
for i in xrange(0, len(large_contours)):
    black_bg = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    cv2.drawContours(black_bg, large_contours, i, color=255, thickness=-1)
    interior_pts = np.where(black_bg == 255)  # collection of sets of coordinates corresponding to the interior points of contours

    for x in xrange(0, len(interior_pts[0])):
        collect_gs[interior_pts[0][x], interior_pts[1][x]] = img_gs[interior_pts[0][x], interior_pts[1][x]]

Image.fromarray(collect_gs).show()

# alternative branch: check the intensities of pixels that were REMOVED. corresponds to white pixels in collect_gs

mask = (collect_gs > 254).astype('uint8')

fig = plt.figure()

a = fig.add_subplot(1, 2, 1)

hist = cv2.calcHist([img_gs], [0], mask, [256], [0, 256])
plt.plot(hist)
print hist
print hist.shape

hist_2 = cv2.calcHist([img_gs], [0], None, [256], [0, 256])
plt.plot(hist_2)
# the histogram of the data
# n, bins, patches = plt.hist(hist, 100, normed=1, facecolor='cyan', alpha=0.82)
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.75) # alpha: intensity (related to transparency)


plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Image:')
# plt.axis([0, 256, 0, 300000]) optional, will get best fit if don't use
plt.grid(True) # whether to show gridlines or not. Default: False

# plt.show()
# then perform global threshold from here

img_for_binarization = collect_gs

# obtain the cuts in horizontal dim through histogram analysis:
foreground = collect_gs < 255
# Image.fromarray(foreground.astype('uint8')*255).show()
Image.fromarray(foreground.astype('uint8')*255).save('binaryimg.png')
# compute the histogram of foreground vs background pixels

# use mpimg
binary = mpimg.imread('binaryimg.png') # collection of lists

fig = plt.figure()

a = fig.add_subplot(1, 2, 2)

print binary
print binary.shape # y, x (in dims)

# want to compute the sum along the y-dim
sum1 = np.sum(binary, axis=1)
print "Sum1: "
print sum1

# find the cutting points based on sum1
t_histo = 130  # number of nonzero pixels in -> direction (threshold value)
potential_cuts = []
for x in xrange(1, len(sum1)):
    if (sum1[x] < 200) and (sum1[x-1] >=200):  # cross downward from the left
        potential_cuts.append(x)  # then is a potential cutting point
print potential_cuts  # collection of y-coordinates corresponding to cuts
print len(potential_cuts)

# get adjustments for the potential cuts: use the next local minimum
# description of local min: f(x) such that f(x-1)>=f(x)<=f(x+1)
adjusted_cuts = []
for value in potential_cuts:
    for x in xrange(0, 100):  # search for the first local min, with x>value. limit search to 300 vals  may have indexerror with this?
        try:
            if (sum1[value+x] <= sum1[value+x+1]) and (sum1[value+x] <= sum1[value+x-1]):  # i.e. if is local min
                adjusted_cuts.append(value+x)
                break
            elif x==99:
                adjusted_cuts.append(value)
        except IndexError:
            adjusted_cuts.append(value)
            break

# next, extract the image regions of the cuts obtained above
# img is in rgb
segments = []
coords = []
if len(adjusted_cuts):
    for x in xrange(0, len(adjusted_cuts)):  # appears to be t
        if x > 0:
            y_low = adjusted_cuts[x-1]
        else:
            y_low = 0

        if x == len(adjusted_cuts)-1:  # to obtain the last (lowest) segment
            y_high = img.shape[0]
        else:
            y_high = adjusted_cuts[x]

        segments.append(collect_gs[y_low:y_high, :])  # contains white background, and have grayscale
        coords.append([y_low, y_high])
        # need to handle the end, i.e. after the last cut
print adjusted_cuts
"""for seg in segments:  # want to collect the histograms of each segment
    # Image.fromarray(seg).show()
    hist_seg = cv2.calcHist([seg], [0], None, [256], [0, 256])  # segment is gs on white background
    plt.plot(hist_seg)"""

# use some existing global thresholding method for each of the segments (different T for each segment)
# inputs should be grayscale (single channel)
black_bg = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

y_cur_min = 0
for seg in zip(segments, coords):  # is in order of increasing y-value
    threshglobal = filters.threshold_li(seg[0])  # Should be an integer? seg[0] refers to the first element, i.e. the segment and not the adjusted cut

    # compute adjustment to the threshold found above based on the ratio of foreground pixels to total pixels in segment
    # background_pixels
    background_pixels = np.where(seg[0] == 255)
    a = len(background_pixels[0])
    seg_img = seg[0]
    b = (seg_img.shape[0])*(seg_img.shape[1])

    adjustment_ratio = 1.0*a/b

    adjustment_value =  35*pow(adjustment_ratio, (1.0/2.0))  # gives a^b
    print "Adjustment: " + str(adjustment_value)
    threshglobal = threshglobal - adjustment_value
    # the larger the ratio, the larger the adjustment effect

    # threshglobal = int(0.9*threshglobal)
    thresholded = seg[0] > threshglobal  # need to stack the images on top of each other after thresholding
    black_bg[seg[1][0]:seg[1][1], :] = thresholded.astype('uint8')*255  # need to have the starting y-coordinate, which should be equal to ending of previous segment
# may want to make adjustments to the threshold value based on the number of background pixels included in the image
# alternatively, if possible, compute the threshold based on the foreground values only. .. but input should be an NxM array. so have to live with the background pixels
Image.fromarray(black_bg).show()
binary = black_bg
with PyTessBaseAPI(psm=6) as api:
    api.SetVariable("tessedit_char_whitelist",
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
    api.SetImage(Image.fromarray(binary.astype('uint8')*255).convert('RGB'))
    print api.GetUTF8Text()
print filename

# next: display the cutting lines for illustrative purposes
"""black = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
black[0:img.shape[0], 0:img.shape[1], 0] = binary.astype('uint8')*255

Image.fromarray(black).show()"""

# may want to remove the small contours again, after this. But may cause problems for the digits that start shattering / disintegrating
# do thresholding within each of the image segments based on difference between bg and fg (lf peak)
# intuitions: the background pixels should be brighter than the foreground pixels. Should have only 1 color of text within one segment (horizontal cut->)

plt.title("Horizontal Frequencies")

plt.plot(sum1) # this gives the histogram for horizontal direction (->), with y varying

# plt.show()

