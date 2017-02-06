#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


from PIL import Image
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from tkFileDialog import askopenfilename
from skimage import measure

"""Implementation of Nagabhushan and Nirmala with a ton of modifications"""

filename = askopenfilename()

img = cv2.imread(filename)

# next split image into the three RGB channels
img_red = img[:, :, 0]
img_green = img[:, :, 1]
img_blue = img[:, :, 2]
print img_red.shape
# perform Canny edge detector on each channel since text may be found in any of the channels -- but which parameters to use?
edge_red = feature.canny(img_red)
edge_green = feature.canny(img_green)
edge_blue = feature.canny(img_blue)
edge_assimilated = np.logical_or(edge_red, np.logical_or(edge_green, edge_blue))  # boolean array

# Next, want to do both horizontal and vertical dilation with 1x3 and 3x1 structuring elements
# Note: paper suggests 1x3 and 3x1, but in our application 5x1 and 1x5 might work better
strel1 = np.zeros((5, 5))
for i in range(0, 5):
    strel1[2][i] = 1

edge_dim_1 = ndi.binary_dilation(edge_assimilated, structure=strel1)

strel2 = np.zeros((5, 5))

for j in range(0, 5):
    strel2[j][2] = 1

edge_dim_2 = ndi.binary_dilation(edge_assimilated, structure=strel2)

edges_dil_comb = np.logical_or(edge_dim_1, edge_dim_2)
all_labels = measure.label(edges_dil_comb, neighbors=8, connectivity=2).astype('uint8')  # same dimensions as image

im2, abac, hierarchy = cv2.findContours(all_labels.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

hierarchy = hierarchy[0]  # get the useful stuff only

# filter contours based on area (reject small ones, which are definitely noise)
large_contours = []
lower_area_bound = 700  # MUST be calibrated so that no text is removed..700 seems appropriate
for contour in zip(abac, hierarchy):
    if cv2.contourArea(contour[0]) > lower_area_bound:  # contour[1][2] > 0
        large_contours.append(contour[0])
print "Number of contours left: " + str(len(large_contours))  # all of these will be processed,so more -> slower
black_bg = np.zeros((img_red.shape[0], img_red.shape[1], 3), dtype='uint8')

# The following two lines may be commented out, they are just for visualizing the contours
cv2.drawContours(black_bg, large_contours, -1, (0, 255, 0), 3)
Image.fromarray(black_bg).show()

# use grayscale intensities to filter: m - k*s. m is mean, s is sd (m, s con component-specific. k is parameter)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # m - k*s will work in grayscale

# find mean, SD for individual connected components


lst_intensities = [] # Initialize empty list

bg_for_final = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')  # for drawing onto
bg_for_final += 255  # change to white background color

for i in range(len(large_contours)):  # loop through the contours 
    cimg_outline = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
    cv2.drawContours(cimg_outline, large_contours, i, color=255, thickness=1)  # Draw the contour outline only 
    pts_outline = np.where(cimg_outline == 255)
    gs_vals_outline = grayscale[pts_outline[0], pts_outline[1]]
    intensity_fg = np.mean(gs_vals_outline)

    cimg_inside = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
    cv2.drawContours(cimg_inside, large_contours, i, color=255, thickness=-1)  # Thickness -1 will fill in the entire contour. i is for drawing individual contours

    # Stats for the entire connected component:
    pts_cp = np.where(cimg_inside == 255)
    mean_cp, sd_cp = cv2.meanStdDev(grayscale[pts_cp[0], pts_cp[1]])

    # Stats for inside (excludes boundary pts) -- may need to exclude more pts later!! (thickness)
    cimg_inside_only = cimg_inside - cimg_outline  # subtract the boundaries from the contours
    pts_inside = np.where(cimg_inside_only == 255)
    gs_vals_inside = grayscale[pts_inside[0], pts_inside[1]]
    intensity_bg = np.mean(gs_vals_inside)

    # Thresholding (want to cvt to binary, and remove non-text pixels)
    if intensity_fg < intensity_bg:  # Note: this part is changed from the paper (inverted, actually)
        k_cp = 0.05
        threshold_cp = mean_cp - (k_cp * sd_cp)
        ret, thresh = cv2.threshold(grayscale.copy(), threshold_cp, 255, cv2.THRESH_BINARY)  # originally THRESH_BINARY_INV
        mask_accepted = cv2.bitwise_and(thresh, cimg_inside)
    else:  
        k_cp = 0.40
        threshold_cp = mean_cp - (k_cp * sd_cp)
        ret, thresh = cv2.threshold(grayscale.copy(), threshold_cp, 255, cv2.THRESH_BINARY_INV)  # originally THRESH_BINARY
        mask_accepted = cv2.bitwise_and(thresh, cimg_inside)

    bg_for_final -= mask_accepted

bg_for_final = 255 - bg_for_final
# Image.fromarray(bg_for_final).show()
# cv2.imwrite('outputcool.png', img & cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR))

# Find the contours (in binary)
im2_out, conts, hierch = cv2.findContours(bg_for_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

accepted_conts = []
largest_area = 2000  # this is just for debugging, to be removed later
for i in range(0, len(conts)):
    area = cv2.contourArea(conts[i])
    if area > 700:
        accepted_conts.append(conts[i])
        if area > largest_area:  # for debugging
            largest_area = area

print "Largest contour area: " + str(largest_area)

bg_for_density_prime = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')

for i in range(0, len(accepted_conts)):
    bg_for_density = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
    cv2.drawContours(bg_for_density, accepted_conts, i, color=255, thickness=-1)
    pts_col = np.where(bg_for_density == 255)
    cv2.drawContours(bg_for_density_prime, accepted_conts, i, color=255, thickness=-1)
    # Get values, both zero and non-zero
    vals_for_numerator = bg_for_final[pts_col[0], pts_col[1]]
    vals_for_denominator = bg_for_density[pts_col[0], pts_col[1]]

    numerator = 0

    for ele in vals_for_numerator:
        if ele > 100:
            numerator += 1

    denominator = len(vals_for_denominator)
    # print denominator

    density = 1.0*numerator/denominator
    # print " Density: " + str(density)
    threshold_density = 0.20  # calibration needed
    # The following part does pretty much nothing so far
    if density > threshold_density:  # this rarely triggers, look into how to apply it 
        bg_for_final[pts_col[0], pts_col[1]] = 0
        print " Triggered, m8! "

# Do comparison: logical_and for the two background images (nonzero)

final_result_I_hopeD = np.logical_and(bg_for_final, bg_for_density_prime)

keep_going = final_result_I_hopeD.astype('uint8')*255  # values 0 or 255 only

# Image.fromarray(keep_going).show()

where_to_look = np.where(keep_going == 255)
gs_mean, gs_sd = cv2.meanStdDev(grayscale[where_to_look[0], where_to_look[1]])
thresh_gs_val = gs_mean + (0.25*gs_sd)
ret, thresh = cv2.threshold(grayscale.copy(), thresh_gs_val, 255, cv2.THRESH_BINARY)  # this is a global threshold with carefully derived threshold
# Image.fromarray(thresh).show()  # is entire image, want overlapping parts only
the_end_mate = np.logical_and(255-thresh, keep_going)  # combine the results from global and local thresholds
Image.fromarray(the_end_mate.astype('uint8')*255).show()
cv2.imwrite('outputbinary.png', 255-the_end_mate.astype('uint8')*255)

# What remains is:
# dilation to thicken the characters
# rotate lines
# try to exploit the global threshold found
# improve algo speed
