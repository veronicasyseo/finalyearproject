#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import matplotlib
matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# import matplotlib.mlab as mlab
# from PIL import Image
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from tkFileDialog import askopenfilename
# from skimage import filters  # alternatively, 'from skimage import filter' (without the s)
from skimage import measure

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
all_labels = measure.label(edges_dil_comb, neighbors=8, connectivity=2).astype('uint8')  # size 2112, 2816 -- values can go up to > 1000 (one label per connected component)
# ^may not be necessary, depending on what function is found for 'bwboundaries'


# cv2 connected components? -- look up
# bwboundaries might be able to do labeling also?
# abac = cv2.findContours(all_labels.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # check if this works


