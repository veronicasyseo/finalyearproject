#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
from PIL import ImageTk, Image
from tkFileDialog import askopenfilename
from tesserocr import PyTessBaseAPI, RIL, iterate_level
import numpy as np
import cv2
from skimage import feature
from skimage import measure
from scipy import ndimage as ndi
import csv
import Levenshtein  # may or may not be used

class simpleapp_tk(Tkinter.Tk):
    def __init__(self, parent):  # constructor
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent  # keep track of our parent
        self.initialize()

    def initialize(self):  # will be used for creating all the widgets
        self.width = 400
        self.height = 400
        self.method = "Hybrid"  # default
        self.grid()  # add things here later

        # button for loading input images
        button1 = Tkinter.Button(self, text=u"Load image !", command=self.OnButtonClickLoad)
        button1.grid(column=0, row=1)

        # button for loading Advance Shipment Notice
        button3 = Tkinter.Button(self, text=u"Load ASN !", command=self.OnButtonClickASN)
        button3.grid(column=2, row=1)

        # label for displaying raw images
        img = np.zeros((self.height, self.width), dtype='uint8')  # default is just a black image
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.label1 = Tkinter.Label(self, image=img)
        self.label1.image = img
        self.label1.grid(column=0, row=0)

        # label for displaying processed image
        self.label2 = Tkinter.Label(self, image=img)
        self.label2.image = img
        self.label2.grid(column=1, row=0)

        # label for displaying text output
        self.displayVar = Tkinter.StringVar()
        self.displayVar.set("OCR output will be displayed here!")
        self.label3 = Tkinter.Label(self, text="OCR output will be displayed here!")  # text vs textvariable
        self.label3.grid(column=2, row=0)

        # label for displaying candidates
        self.displayCands = Tkinter.StringVar()
        self.displayCands.set("Candidates will be displayed here! (to be implemented)")
        self.label4 = Tkinter.Label(self, textvariable=self.displayCands)
        self.label4.grid(column=0, row=2)

        # dropdown menu for selecting image processing procedure
        optionList = ["Hybrid", "Advanced", "Basic", "ContourGaussianKernelOtsu", "GaussianKernelAndOtsu", "Otsu"]  # add more later
        self.dropVar = Tkinter.StringVar()
        self.dropVar.set("Hybrid")  # default
        self.dropMenu1 = Tkinter.OptionMenu(self, self.dropVar, *optionList, command=self.func)
        self.dropMenu1.grid(column=1, row=1)

        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)
        self.update()
        self.geometry(self.geometry())

    def OnButtonClickLoad(self):
        filename = askopenfilename()  # prompts user to choose a file
        self.filename = filename
        img2 = Image.open(filename)  # what if user cancelled?
        img2 = img2.resize((self.width, self.height), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img2)

        self.label1.config(image=img2)
        self.label1.image = img2
        processor = img_processor(self.filename)
        if self.method in "Basic":
            text, img = processor.Benchmark()  # will later update to use the method chosen in drop-down menu
            self.DisplayProcessed(img.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)  # implement later
        elif self.method in "Advanced":
            text, img = processor.Advanced()
            self.DisplayProcessed(img.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
        elif self.method in "Otsu":
            text, img = processor.Otsu()
            self.DisplayProcessed(img.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
        elif self.method in "GaussianKernelAndOtsu":
            text, img = processor.GaussianKernelAndOtsu()
            self.DisplayProcessed(img.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
        elif self.method in "ContourGaussianKernelOtsu":
            text, img = processor.ContourGaussianKernelOtsu()
            self.DisplayProcessed(img.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
        elif self.method in "Hybrid":
            text, img = processor.Hybrid()
            self.DisplayProcessed(img.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
        else:
            print "Method selection error!"

    def OnButtonClickASN(self):
        CSVName = askopenfilename()
        loader = CSVLoader(CSVName)
        asn_csv_array = loader.Load()

    def DisplayProcessed(self, image_processed):
        image_processed = ImageTk.PhotoImage(image_processed)
        self.label2.config(image=image_processed)
        self.label2.image = image_processed

    def func(self, value):
        self.method = value
        # print self.method

    def DisplayOCRText(self, text):
        if len(text) > 200:
            text = "Use other image processing methods!"
        self.label3.config(text=text)

    def DisplayCands(self, text):
        pass  # to be implemented 

class img_processor():
    def __init__(self, filename):
        self.filename = filename

    def Benchmark(self):
        """Simple OCR with Tesseract only"""
        with PyTessBaseAPI(psm=6) as api:
            api.SetVariable("tessedit_char_whitelist",
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
            api.SetImageFile(self.filename)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()
            return text_output, image_processed

    def Basic(self, thresholded):
        with PyTessBaseAPI(psm=6) as api:
            api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
            api.SetImage(thresholded)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()
            return text_output, image_processed

    def Hybrid(self):
        img = cv2.imread(self.filename)

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
        all_labels = measure.label(edges_dil_comb, neighbors=8, connectivity=2).astype(
            'uint8')  # same dimensions as image

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
        # Image.fromarray(black_bg).show()

        # use grayscale intensities to filter: m - k*s. m is mean, s is sd (m, s con component-specific. k is parameter)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # m - k*s will work in grayscale

        # find mean, SD for individual connected components

        lst_intensities = []  # Initialize empty list

        bg_for_final = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')  # for drawing onto
        bg_for_final += 255  # change to white background color

        for i in range(len(large_contours)):  # loop through the contours
            cimg_outline = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
            cv2.drawContours(cimg_outline, large_contours, i, color=255, thickness=1)  # Draw the contour outline only
            pts_outline = np.where(cimg_outline == 255)
            gs_vals_outline = grayscale[pts_outline[0], pts_outline[1]]
            intensity_fg = np.mean(gs_vals_outline)

            cimg_inside = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
            cv2.drawContours(cimg_inside, large_contours, i, color=255,
                             thickness=-1)  # Thickness -1 will fill in the entire contour. i is for drawing individual contours

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
                ret, thresh = cv2.threshold(grayscale.copy(), threshold_cp, 255,
                                            cv2.THRESH_BINARY)  # originally THRESH_BINARY_INV
                mask_accepted = cv2.bitwise_and(thresh, cimg_inside)
            else:
                k_cp = 0.40
                threshold_cp = mean_cp - (k_cp * sd_cp)
                ret, thresh = cv2.threshold(grayscale.copy(), threshold_cp, 255,
                                            cv2.THRESH_BINARY_INV)  # originally THRESH_BINARY
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

            density = 1.0 * numerator / denominator
            # print " Density: " + str(density)
            threshold_density = 0.20  # calibration needed
            # The following part does pretty much nothing so far
            if density > threshold_density:  # this rarely triggers, look into how to apply it
                bg_for_final[pts_col[0], pts_col[1]] = 0
                print " Triggered, m8! "

        # Do comparison: logical_and for the two background images (nonzero)

        final_result_I_hopeD = np.logical_and(bg_for_final, bg_for_density_prime)

        keep_going = final_result_I_hopeD.astype('uint8') * 255  # values 0 or 255 only

        # Image.fromarray(keep_going).show()

        where_to_look = np.where(keep_going == 255)
        gs_mean, gs_sd = cv2.meanStdDev(grayscale[where_to_look[0], where_to_look[1]])
        thresh_gs_val = gs_mean + (0.25 * gs_sd)
        ret, thresh = cv2.threshold(grayscale.copy(), thresh_gs_val, 255,
                                    cv2.THRESH_BINARY)  # this is a global threshold with carefully derived threshold
        # Image.fromarray(thresh).show()  # is entire image, want overlapping parts only
        the_end_mate = np.logical_and(255 - thresh, keep_going)  # combine the results from global and local thresholds
        the_end_image = Image.fromarray(the_end_mate.astype('uint8') * 255)
        # the_end_image.show()
        # cv2.imwrite('outputbinary.png', 255 - the_end_mate.astype('uint8') * 255)

        img_ret, text_ret = self.Basic(the_end_image)

        return img_ret, text_ret

    def Advanced(self):
        """Does contour analysis to find the likely text region(s) of an image,
         uses Otsu's binarization on the result
          and performs OCR with Tesseract"""
        # load image:
        img_raw = cv2.imread(self.filename)

        # convert to grayscale
        img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  # compare BGR and RGB -- results appear to be the same
        binary = img_grey.copy()  # change name later

        # convert to float32
        img_grey = img_grey.astype('float32')
        img_grey /= 255  # normalize

        dct = cv2.dct(img_grey)  # discrete cosine transform of the grayscale image
        vr = 1.
        hr = .95
        dct[0:vr * dct.shape[0], 0:hr * dct.shape[1]] = 0  # warning can be ignored
        gray = cv2.idct(dct)  # inverse discrete cosine transform
        gray = cv2.normalize(gray, -1, 0, 1, cv2.NORM_MINMAX)
        gray *= 255
        gray = gray.astype('uint8')

        # prepare the image for finding contours
        gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),  # Originally (15,15), used 24,24
                                iterations=1)
        gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),  # Originally (11,11)
                                iterations=1)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

        abac, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxmask = np.zeros(gray.shape, gray.dtype)  # just a black background of same size and type as the image

        contlist = []  # To be used for larger (merged) contours

        self.disthreshold = 10  # This is currently creating large bias towards the bottom-right corner of the image.

        lower = 0.0001  # probably needs adjusting
        upper = 0.80  # may need adjustments
        scale = 1.08  # adjust

        # discard some contours (noise) based on area and resize them slightly (scaling)
        for i in xrange(len(contours)):
            if upper > cv2.contourArea(
                    contours[i]) / img_raw.size > lower:
                x, y, w, h = cv2.boundingRect(contours[i])
                x_left = int(x * (2 - scale))
                x_right = int((x + w) * scale)
                y_top = int(y)
                y_bottom = int(y + h)
                contpts = [y_top, x_left, y_bottom, x_right]
                contlist.append(contpts)
        contlist.sort()

        merged = self.proximitycheck(contlist)  # merge contours

        for cnt in merged:
            y_top = cnt[0]
            x_left = cnt[1]
            y_bottom = cnt[2]
            x_right = cnt[3]
            res = img_raw[y_top:y_bottom, x_left:x_right]
            res = Image.fromarray(res)  # check if necessary

            cv2.rectangle(boxmask, (x_left, y_top), (x_right, y_bottom), color=255,
                          thickness=-1)

        cv2.imwrite('output.png', img_raw & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))
        # cv2.imwrite('ingray.png', binary & boxmask)

        # Need to do some operations on the image before feeding it to OCR in order to improve accuracy:
        ocr_img = cv2.imread('output.png')
        ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)

        # Do Otsu binarization on the image:
        ocr_otsu = cv2.threshold(ocr_gray, 0, 255, cv2.THRESH_OTSU)[1]
        ocr_ready = Image.fromarray(ocr_otsu)

        # Done with contour analysis (for now)
        # Finally, feed the whole processed image
        with PyTessBaseAPI(psm=6) as api:
            api.SetImage(ocr_ready)
            api.SetVariable("tessedit_char_whitelist",
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
            # handle text output
            output_text = api.GetUTF8Text().encode(
                "utf-8")  # Encode as utf-8, otherwise would be ascii by python's default
            f = open("output.txt", 'w')
            f.write(output_text)  # overwrites
            f.close()  # close file to release memory
            api.GetThresholdedImage().show()  # optional, for debugging

            # draw rectangles to show where Tesseract found characters and lines
            ls = api.GetTextlines()
            img = cv2.imread(self.filename)

            for line in ls:  # draw all the rectangles
                x = line[1]['x']
                y = line[1]['y']
                w = line[1]['w']
                h = line[1]['h']
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # draw red rectangles around text lines

            iterator = api.GetIterator()
            iterator.Begin()
            level = RIL.SYMBOL
            for r in iterate_level(iterator, level):
                # print r.BoundingBox(level)
                x = r.BoundingBox(level)[0]
                y = r.BoundingBox(level)[1]
                x_2 = r.BoundingBox(level)[2]
                y_2 = r.BoundingBox(level)[3]

                img = cv2.rectangle(img, (x, y), (x_2, y_2), (0, 255, 0),
                                    3)  # Draw a green rectangle around each character found by OCR

            output_img = Image.fromarray(img)
            # want to also destroy the iterator to clear memory (if needed)
        return output_text, output_img

    def ContourGaussianKernelOtsu(self):
        # load image:
        img_raw = cv2.imread(self.filename)

        # convert to grayscale
        img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        binary = img_grey.copy()  # change name later

        # convert to float32
        img_grey = img_grey.astype('float32')
        img_grey /= 255  # normalize

        dct = cv2.dct(img_grey)  # discrete cosine transform of the grayscale image
        vr = 1.
        hr = .95
        dct[0:vr * dct.shape[0], 0:hr * dct.shape[1]] = 0  # warning can be ignored
        gray = cv2.idct(dct)  # inverse discrete cosine transform
        gray = cv2.normalize(gray, -1, 0, 1, cv2.NORM_MINMAX)
        gray *= 255
        gray = gray.astype('uint8')

        # prepare the image for finding contours
        gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                                # Originally (15,15), used 24,24
                                iterations=1)
        gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),  # Originally (11,11)
                                iterations=1)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

        abac, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxmask = np.zeros(gray.shape, gray.dtype)  # just a black background of same size and type as the image

        contlist = []  # To be used for larger (merged) contours

        self.disthreshold = 10  # This is currently creating large bias towards the bottom-right corner of the image.

        lower = 0.0001  # probably needs adjusting
        upper = 0.80  # may need adjustments
        scale = 1.08  # adjust

        # discard some contours (noise) based on area and resize them slightly (scaling)
        for i in xrange(len(contours)):
            if upper > cv2.contourArea(
                    contours[i]) / img_raw.size > lower:
                x, y, w, h = cv2.boundingRect(contours[i])
                x_left = int(x * (2 - scale))
                x_right = int((x + w) * scale)
                y_top = int(y)
                y_bottom = int(y + h)
                contpts = [y_top, x_left, y_bottom, x_right]
                contlist.append(contpts)
        contlist.sort()

        merged = self.proximitycheck(contlist)  # merge contours

        for cnt in merged:
            y_top = cnt[0]
            x_left = cnt[1]
            y_bottom = cnt[2]
            x_right = cnt[3]
            res = img_raw[y_top:y_bottom, x_left:x_right]
            res = Image.fromarray(res)  # check if necessary

            cv2.rectangle(boxmask, (x_left, y_top), (x_right, y_bottom), color=255,
                          thickness=-1)

        cv2.imwrite('output.png', img_raw & cv2.cvtColor(boxmask, cv2.COLOR_GRAY2BGR))
        # cv2.imwrite('ingray.png', binary & boxmask)

        # Need to do some operations on the image before feeding it to OCR in order to improve accuracy:
        ocr_img = cv2.imread('output.png')
        ocr_gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)

        image_raw = cv2.imread('output.png')
        image_grayscale = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)  # check if has to be grayscale
        blur = cv2.GaussianBlur(image_grayscale, (21, 21), 0)
        img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        img_for_tess = Image.fromarray(img_binary)
        img_for_tess.show()

        # Tesseract part
        with PyTessBaseAPI(psm=6) as api:
            api.SetImage(img_for_tess)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()
            print text_output  # to be removed
        return text_output, image_processed

    def GaussianKernelAndOtsu(self):
        image_raw = cv2.imread(self.filename)
        image_grayscale = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)  # check if has to be grayscale
        blur = cv2.GaussianBlur(image_grayscale, (21, 21), 0)
        img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        img_for_tess = Image.fromarray(img_binary)
        img_for_tess.show()

        # Tesseract part
        with PyTessBaseAPI(psm=6) as api:
            api.SetImage(img_for_tess)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()
            print text_output  # to be removed
        return text_output, image_processed

    def Otsu(self):
        """Uses only Otsu's binarization, followed by Tesseract for OCR"""
        # load image
        img_raw = cv2.imread(self.filename)
        img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)[1]
        img_for_tess = Image.fromarray(img_binary)
        # Tesseract part
        with PyTessBaseAPI(psm=6) as api:
            api.SetImage(img_for_tess)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()

        return text_output, image_processed

    def overlap(self, row1, row2):  # Note: will have to check against contlist also, to see if overlap
        """Helper for contour analysis
        Returns true if bounding rectangles of contours overlap"""
        if row1[2] + 0.3 * self.disthreshold >= row2[0] and (
                row2[1] <= row1[3] + self.disthreshold or row2[1] <= row1[1] + self.disthreshold):
            # row1[2]+disthreshold >= row2[0] and (row2[1] <= row1[3]+disthreshold or row2[1] <= row1[1]+disthreshold)
            return True  # check the above conditions...Part 1 Correct
        else:
            return False

    def proximitycheck(self, matrix):
        """Returns a list of contours after merging close and overlapping contours.
        May need special case for when the matrix (input) has only one element"""
        output = []
        while len(matrix) > 0:  # Keep doing until no more elements in the original matrix
            done = False  # done is for the current i-value
            if len(output) == 0:
                output.append(matrix.pop(0))
            elif len(matrix) == 1:
                # check if can match to any of the elements in the output array, otherwise simply append it
                row1 = matrix[0]
                row2 = output[len(output) - 1]  # the last element. Need error handling (possibly)
                if self.overlap(row1, row2):
                    x_min = min((row1[1], row2[1]))
                    y_min = min((row1[0], row2[0]))
                    x_max = max((row1[3], row2[3]))
                    y_max = max((row1[2], row2[2]))
                    matrix.pop(0)
                    row4 = [y_min, x_min, y_max, x_max]
                    output.append(row4)
            else:
                # Do comparison within the input matrix
                row1 = matrix[0]  # To be compared with all the other elements until match is found.

                for j in range(0, len(
                        matrix)):  # Need to deal with out of bound- does not break out of loop properly...so the length of the array changes while iterating
                    if not done and j < len(matrix):  # avoid out of bound
                        row2 = matrix[j]
                        if self.overlap(row1, row2):
                            x_min = min((row1[1], row2[1]))
                            y_min = min((row1[0], row2[0]))
                            x_max = max((row1[3], row2[3]))
                            y_max = max((row1[2], row2[2]))
                            matrix.pop(j)
                            matrix.pop(0)
                            # Should check if overlap with any elements in the output matrix first # output.append(y_min, x_min, y_max, x_max)
                            row3 = [y_min, x_min, y_max, x_max]
                            # Start with len(output)-1 and iterate at most 5 times
                            # if len(output) > 0:
                                # print output
                            for l in range(0, len(output)):  # Need to deal with out of bound issues
                                if done:
                                    break
                                elif self.overlap(row3, output[max(len(output) - l - 1, 0)]):
                                    r = max(len(output) - l - 1, 0)
                                    x_min = min(row3[1], output[r][1])
                                    y_min = min(row3[0], output[r][0])
                                    x_max = max(row3[3], output[r][3])
                                    y_max = max(row3[2], output[r][2])
                                    output.pop(r)
                                    row4 = [y_min, x_min, y_max, x_max]
                                    output.append(row4)
                                    done = True

                            if not done:  # if does not match any of the other elements in output matrix
                                output.append(row3)
                                done = True
                    if not done:  # CHECK: Indenting
                        output.append(matrix.pop(0))
                        done = True

        return output  # Boolean

class CSVLoader():
    """Loads the ASN file from csv format, and stores all the values in an array"""
    def __init__(self, ASNfile):
        self.ASNfile = ASNfile

    def Load(self):
        """Load and return the array"""
        asn_data = []

        with open(self.ASNfile, "rb") as csvfile:  # load the ASN from csv file, store in array (line by line)
            spamreader = csv.reader(csvfile, delimiter=';',
                                    quotechar='|')  # Spamreader is an object..need to save the info somehow

            for row in spamreader:  # Row only contains 1 row at any given time
                print ', '.join(row)
                asn_data.append(row)
        csvfile.close()

        return asn_data

class outputInterpreter():
    def __init__(self, OCR_output):
        self.OCR_output = OCR_output
        # may or may not need to use string.encode('utf-8')

    def subString6Digit(self, input_string):
        """Check if have at least 6 digits in a row, return true if yes"""
        digits = False
        for x in range(6, len(input_string)):
            if input_string[x-6:x].isdigit():
                digits = True
                break
        return digits

    def isItemCode(self, text_line):
        """Should be either the first or second line normally. Consist of mainly digits"""
        # find the general form of the text_line, see if any substring matches the format of item code
        # translate the string, and check if have a long substring of digits, which will indicate it's an item code.
        input_string = text_line.strip('\n')  # get rid of the end-line character
        input_string = input_string.strip(" ")
        input_string = input_string.upper()
        input_string = input_string.replace("O", "0")
        input_string = input_string.replace("L", "1")
        input_string = input_string.translate(None, '-_() ')

        if input_string.isdigit():
            return True
        elif self.subString6Digit(input_string):
            return True
        else:
            return False

    def isSolidCode(self, input_string):
        """Digits mainly, lots of zeros. If handwritten, then will be short and nonsense
        Check if have triple 0s, allowing o and O. Consider allowing 6"""
        input_string = input_string.strip('\n')
        input_string = input_string.strip(" ")
        input_string = input_string.upper()
        input_string = input_string.replace("O", "0")
        input_string = input_string.translate(None, '-_() ')

        if "00" in input_string:  # consider adding more cases
            return True
        else:
            return False

    def isAssortCode(self):
        """One char followed by 3 digits"""
        pass

    def isDescription(self):
        """Lots of characters"""
        pass

    def isQuantity(self, input_string):
        """Contains hints like QTY, PCS"""
        pass

    def isCartonNo(self):
        """Has some indicators..."""
        pass

    def isNothing(self):
        pass

    def categorizeLines(self):
        # Want to store the line indexes of each type of line
        categories = {'Itemcode': False, 'Solidcode': False, "Assortcode": False, 'Quantity': False, 'Cartonnumber': False, 'Description': False, 'Nothing': False}

        # split the string around "\n", and store the result in an array
        text_array = self.OCR_output.split(sep="\n")  # want to use the newline character as separator
        while not categories['Itemcode']:  # Need to make sure the loop stops once Itemcode becomes a digit
            # what if there is no line categorized as item code????
            for i in range(0, len(text_array)): # will be a long string rather than an array of string (separated by newline character)
                if self.isItemCode(text_array[i]):  # don't need to check after have found the line corresponding to item code (save comp time)
                    # can use while loop
                    categories['Itemcode'] = i  # store the index of the item code
                    break
                if i == len(categories) and not ['Itemcode']:
                    print "Unable to find Itemcode, we have a problem m8!"
                    break
        # now only consider lines with index > i (assume no sensible output is above the item code
        # May have error if i == len(text_array), but in this case we're screwed anyways
        for j in range(i+1, len(text_array)-1):  # next LF solid/assort code and try to categorize
            """if j+1 contains "PCS", then most likely line j is our solid/assort code, regardless of the ugliness of line j's text
            solid code is more common than assort code, so first LF solid"""
            if self.isSolidCode(text_array[j]):
                categories['Solidcode'] = j
                break
        # consider maximum allowable number of lines checked: should not have too much noise
        # assume absense of solid code means we have assort code?
        # but: in some cases messy text comes out


class matcher():
    def __init__(self, csv_array):
        self.csv_array = csv_array

if __name__ == "__main__":
    app = simpleapp_tk(None)  # No parent because first element
    app.title('HKUST x ABC Company')
    app.mainloop()  # Run infinite loop, waiting for events.
