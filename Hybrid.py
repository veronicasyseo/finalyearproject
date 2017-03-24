#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

from PIL import Image
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
import time
from tesserocr import PyTessBaseAPI, iterate_level, RIL

"""Implementation of Nagabhushan and Nirmala with a ton of modifications"""

def Basic(thresholded, img_array):
    with PyTessBaseAPI(psm=6) as api:  # originally psm 6. Tried psm=4 without too good results.
        api.SetVariable("tessedit_char_whitelist",
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
        api.SetImage(thresholded)
        text_output = api.GetUTF8Text().encode('utf-8')
        print text_output  # can remove
        iterator = api.GetIterator()
        iterator.Begin()
        level = RIL.TEXTLINE
        boxes = []
        for r in iterate_level(iterator, level):
            boxes.append(r.BoundingBox(level))
        print boxes
        img_from_tess = img_array.copy()

        iterator = api.GetIterator()
        iterator.Begin()
        level = RIL.SYMBOL
        for r in iterate_level(iterator, level):
            try:
                # print r.BoundingBox(level)
                x = r.BoundingBox(level)[0]
                y = r.BoundingBox(level)[1]
                x_2 = r.BoundingBox(level)[2]
                y_2 = r.BoundingBox(level)[3]

                img_from_tess = cv2.rectangle(img_from_tess, (x, y), (x_2, y_2), 255,
                                              3)  # Draw a rectangle around each character found by OCR
            except TypeError:
                print "Don't have iterator!"

        boxed = Image.fromarray(img_from_tess)
        # boxed.show()
        boxed.save('boxed.png')
        # instead of returning the iterator, want to extract slices corresponding to textlines and pass them on.
        # hopefully, the list will be ordered, for easier use in the following
    return text_output, boxes


def Hybrid(filename):
    # filename = askopenfilename()

    img = cv2.imread(filename)

    tic = time.clock()

    contour_area_min = 350  # set a fixed threshold for the contour area (noise removal)

    # next split image into the three RGB channels
    img_red = img[:, :, 0]
    img_green = img[:, :, 1]
    img_blue = img[:, :, 2]
    print img_red.shape
    # perform Canny edge detector on each channel since text may be found in any of the channels -- but which parameters to use?
    edge_red = cv2.Canny(img_red, 50, 100)
    edge_green = cv2.Canny(img_green, 50, 100)
    edge_blue = cv2.Canny(img_blue, 50, 100)
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

    all_labels = np.logical_or(edge_dim_1, edge_dim_2).astype('uint8')

    im2, abac, hierarchy = cv2.findContours(all_labels.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    hierarchy = hierarchy[0]  # get the useful stuff only

    large_contours = []
    largest_contour_area = 0
    for contour in zip(abac, hierarchy):
        if cv2.contourArea(contour[0]) > largest_contour_area:
            largest_contour_area = cv2.contourArea(contour[0])
        if cv2.contourArea(contour[0]) > contour_area_min:  # contour[1][2] > 0
            large_contours.append(contour[0])
    print "Number of contours left: " + str(len(large_contours))  # all of these will be processed,so more -> slower
    black_bg = np.zeros((img_red.shape[0], img_red.shape[1], 3), dtype='uint8')
    print "Largest contour area: " + str(largest_contour_area)
    # The following two lines may be commented out, they are just for visualizing the contours
    cv2.drawContours(black_bg, large_contours, -1, (0, 255, 0), 3)

    # use grayscale intensities to filter: m - k*s. m is mean, s is sd (m, s con component-specific. k is parameter)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # m - k*s will work in grayscale

    bg_for_final = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')  # for drawing onto
    bg_for_final += 255  # change to white background color

    for i in range(len(large_contours)):  # loop through the contours
        cimg_outline = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
        cv2.drawContours(cimg_outline, large_contours, i, color=255, thickness=1)  # Draw the contour outline only
        pts_outline = np.where(cimg_outline == 255)
        gs_vals_outline = grayscale[pts_outline[0], pts_outline[1]]
        intensity_fg = np.mean(gs_vals_outline)  # what is this used for?

        cimg_inside = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
        cv2.drawContours(cimg_inside, large_contours, i, color=255,
                         thickness=-1)  # Thickness -1 will fill in the entire contour. i is for drawing individual contours

        # Stats for the entire connected component:
        pts_cp = np.where(cimg_inside == 255)
        mean_cp, sd_cp = cv2.meanStdDev(grayscale[pts_cp[0], pts_cp[1]])

        # Stats for inside (excludes boundary pts) -- may need to exclude more pts later!! (thickness)
        # cimg_inside_only = cimg_inside - cimg_outline  # subtract the boundaries from the contours
        # pts_inside = np.where(cimg_inside_only == 255)
        # gs_vals_inside = grayscale[pts_inside[0], pts_inside[1]]
        # intensity_bg = np.mean(gs_vals_inside)

        # Thresholding (want to cvt to binary, and remove non-text pixels)
        if intensity_fg < mean_cp:  # Note: this part is changed from the paper (inverted, actually)
            k_cp = 0.05
            threshold_cp = mean_cp - (k_cp * sd_cp)
            ret, thresh = cv2.threshold(grayscale.copy(), threshold_cp, 255,
                                        cv2.THRESH_BINARY)  # originally THRESH_BINARY_INV
            mask_accepted = cv2.bitwise_and(thresh, cimg_inside)
        else:
            k_cp = 0.40  # was OG 0.40. Smaller value appears to introduce more noise
            threshold_cp = mean_cp - (k_cp * sd_cp)
            ret, thresh = cv2.threshold(grayscale.copy(), threshold_cp, 255,
                                        cv2.THRESH_BINARY_INV)  # originally THRESH_BINARY
            mask_accepted = cv2.bitwise_and(thresh, cimg_inside)
            # debugging: interested in knowing whether this part is triggered at all
            # print "Triggered: the outline intensity is larger than the inside intensity (in grayscale)"

        bg_for_final -= mask_accepted

    bg_for_final = 255 - bg_for_final
    # Image.fromarray(bg_for_final).show()  # snow problem already existing at this stage
    # cv2.imwrite('outputcool.png', img & cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR))

    # Find the contours (in binary)
    im2_out, conts, hierch = cv2.findContours(bg_for_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    accepted_conts = []
    # largest_area = 2000  # this is just for debugging, to be removed later

    for i in range(0, len(conts)):
        area = cv2.contourArea(conts[i])
        if area > contour_area_min:
            accepted_conts.append(conts[i])
            # if area > largest_area:  # for debugging
                # largest_area = area

    # print "Largest contour area: " + str(largest_area)

    bg_for_density_prime = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
    cv2.drawContours(bg_for_density_prime, accepted_conts, -1, color=255, thickness=-1)

    # Do comparison: logical_and for the two background images (nonzero)

    final_result_I_hopeD = np.logical_and(bg_for_final, bg_for_density_prime)  # may be gone if changes are made above

    keep_going = final_result_I_hopeD.astype('uint8') * 255  # values 0 or 255 only

    # Image.fromarray(keep_going).show()

    where_to_look = np.where(keep_going == 255)
    gs_mean, gs_sd = cv2.meanStdDev(grayscale[where_to_look[0], where_to_look[1]])
    thresh_gs_val = gs_mean + (0.75 * gs_sd)  # need to calibrate. Tried: +0.25, +0.45, +0.75, +1.10
    # +1.10 too high, reintroduces noise 3
    # +0.75 too low, but at least minimal noise only reintroduced. -> dilation
    ret, thresh = cv2.threshold(grayscale.copy(), thresh_gs_val, 255,
                                cv2.THRESH_BINARY)  # this is a global threshold with carefully derived threshold
    # Image.fromarray(thresh).show()  # is entire image, want overlapping parts only
    the_end_mate = np.logical_and(255 - thresh, keep_going)  # combine the results from global and local thresholds

    print "Shape of didn't think so"
    the_end_image = Image.fromarray(the_end_mate.astype('uint8') * 255)
    did_not_think_so = 255 - the_end_mate.astype('uint8')*255
    print did_not_think_so.shape
    Image.fromarray(did_not_think_so).show  # why is this not displayed?
    # the_end_image.show()
    # cv2.imwrite('outputbinary.png', 255 - the_end_mate.astype('uint8') * 255)
    # Added steps: To resolve the "cloud" issues:
    nonzero_count = np.count_nonzero(did_not_think_so)
    nonzero_threshold = 0.90  # not appropriate threshold
    img_size = img_red.shape[0] * img_red.shape[1]

    if (1.0*nonzero_count/img_size) < nonzero_threshold:  # could also try gaussian blur? This is not appropriate for detecting problems
        print "Too many black pixels, need to do reprocessing!"
        print "Nonzero pixels: " + str(nonzero_count)
        print "Image size: " + str(img_size)

        can_red = feature.canny(img_red, sigma=3)  # alternatively, add lower and upper thresholds of 50/100 TBC
        can_green = feature.canny(img_green, sigma=3)
        can_blue = feature.canny(img_blue, sigma=3)

        # use logical OR operator to combine the results of canny
        combined_canny = np.logical_or(can_red, np.logical_or(can_green, can_blue)).astype('uint8')*255  # should be boolean

        # dilation turns out to be useful
        strel1 = np.zeros((5, 5))
        for i in range(0, 5):
            strel1[2][i] = 1
        # horizontal dilation
        edge_dim_1 = ndi.binary_dilation(combined_canny, structure=strel1)

        strel2 = np.zeros((5, 5))

        for j in range(0, 5):
            strel2[j][2] = 1
        # vertical dilation
        edge_dim_2 = ndi.binary_dilation(combined_canny, structure=strel2)
        # combine the results with OR operator
        edges_dil_comb = np.logical_or(edge_dim_1, edge_dim_2)
        # all_labels = measure.label(edges_dil_comb, neighbors=8, connectivity=2).astype('uint8')  # same dimensions as image
        all_labels = edges_dil_comb.astype('uint8')
        # Image.fromarray(all_labels*255).show()

        # contour analysis -- only needed in order to find the positions of the structures
        im2_spec, contours_spec, hierarchy_spec = cv2.findContours(all_labels*255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        large_contours_spec = []
        for cont in contours_spec:
            if cv2.contourArea(cont) > contour_area_min:  # never passed, likely due to the area function failing
                large_contours_spec.append(cont)

        # next, deal with each contour individually
        black_bg_spec = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
        black_bg_spec += 255
        print "Before area req: " + str(len(contours_spec))
        print "After: " + str(len(large_contours_spec))

        for x in range(0, len(large_contours_spec)):
            # start with outline
            outline_spec = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
            cv2.drawContours(outline_spec, large_contours_spec, x, color=255, thickness=1)
            pts_outline_spec = np.where(outline_spec == 255)
            gs_outline_spec = grayscale[pts_outline_spec[0], pts_outline_spec[1]]  # collect the gs values of the outline of the current contour
            intensity_fg_spec = np.mean(gs_outline_spec)

            # next do the interior points
            interior_spec = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
            cv2.drawContours(interior_spec, large_contours_spec, x, color=255, thickness=-1)
            pts_interior_spec = np.where(interior_spec == 255)
            gs_interior_spec = grayscale[pts_interior_spec[0], pts_interior_spec[1]]
            intensity_bg_spec, sd_spec = cv2.meanStdDev(gs_interior_spec)

            if intensity_fg_spec < intensity_bg_spec:
                k_cp_spec = 0.05
                threshold = intensity_bg_spec - (k_cp_spec * sd_spec)
                # perform binary thresholding
                ret, thresh_spec = cv2.threshold(grayscale.copy(), threshold, 255, cv2.THRESH_BINARY)
                # use the result to create a mask
                mask_accepted_spec = cv2.bitwise_and(thresh_spec, interior_spec)
            else:
                k_cp_spec = 0.40
                threshold = intensity_bg_spec - (k_cp_spec * sd_spec)
                # perform binary thresholding
                ret, thresh_spec = cv2.threshold(grayscale.copy(), threshold, 255, cv2.THRESH_BINARY_INV)
                # mask w/ res
                mask_accepted_spec = cv2.bitwise_and(thresh_spec, interior_spec)
                # debugging: interested in knowing whether this part is triggered at all
                # print "Triggered: the outline intensity is larger than the inside intensity (in grayscale)"

            black_bg_spec -= mask_accepted_spec
        bg_for_final_spec = 255 - black_bg_spec
        # Image.fromarray(bg_for_final_spec).show()

        # contour analysis again, to be followed by global thresholding on the grayscale image
        im2_k, contours_k, hierarchy_k = cv2.findContours(bg_for_final_spec, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # the hierarchy does not matter

        big_contours_k = []

        for cnt_k in contours_k: # simple area thresholding again to remove the noise
            if cv2.contourArea(cnt_k) > contour_area_min:
                big_contours_k.append(cnt_k)

        black_bg_k = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
        cv2.drawContours(black_bg_k, big_contours_k, -1, color=255, thickness=-1)

        combine_spec_k = np.logical_and(black_bg_k, bg_for_final_spec)
        combine_unsigned = combine_spec_k.astype('uint8')*255
        regions_in_both = np.where(combine_unsigned == 255)
        mean_k, sd_k = cv2.meanStdDev(grayscale[regions_in_both[0], regions_in_both[1]])
        threshold_k = mean_k + (2.0*sd_k)  # much larger threshold than in the OG part

        ret, thresholded_k = cv2.threshold(grayscale.copy(), threshold_k, 255, cv2.THRESH_BINARY)
        the_end_mate = np.logical_and(255-thresholded_k, combine_unsigned)

        print "Reprocessing completed! "
        # need to merge results with the
    # Image.fromarray(did_not_think_so).show()
    img_ret = 255 - the_end_mate.astype('uint8')*255  # black text on white background
    Image.fromarray(img_ret).save("Hybrid.png")  # black text on white background

    the_end_image = Image.fromarray(the_end_mate.astype('uint8') * 255)
    did_not_think_so = 255 - the_end_mate.astype('uint8')*255

    # attempt to use dilation to thicken the characters
    dil_size = 5
    strel1 = np.zeros((dil_size, dil_size), dtype='uint8')  # dilation is affecting NONZERO elements, need to have white text and black bg
    strel2 = np.zeros((dil_size, dil_size), dtype='uint8')

    for k in range(0, dil_size):
        strel1[k][2] = 1
        # strel2[1][k] = 1

    for_dilation = the_end_mate
    dilated = ndi.binary_dilation(for_dilation, structure=strel1)
    # dilated = ndi.binary_dilation(dilated, structure=strel2)

    for_tess_arr = dilated.astype('uint8')*255  # looks promising, need to test on Tess
    for_tess_img = Image.fromarray(for_tess_arr)
    toc = time.clock()
    time_taken = toc - tic

    print "Image processing time: " + str(time_taken)

    # text_ret, boxes = self.Basic(the_end_image, did_not_think_so)
    text_ret, boxes = Basic(for_tess_img, for_tess_arr)

    return text_ret, boxes, img_ret
