#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
import csv
import time
from tesserocr import PyTessBaseAPI, RIL, iterate_level
from tkFileDialog import askopenfilename, askdirectory

import Levenshtein  # may or may not be used
import cv2
import numpy as np
from PIL import ImageTk, Image
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import threshold_adaptive
import re
import collections
import pprint
import os
import pickle
import Segmentation  # requires you to download Segmentation.py from our github repository
from random import shuffle
import math

class simpleapp_tk(Tkinter.Tk):
    def __init__(self, parent):  # constructor
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent  # keep track of our parent
        self.initialize()
        self.asn_data = []
        self.match_instance = 0
        self.ASN_loaded = False

    def setMatcher(self, matcher_instance):
        self.match_instance = matcher_instance

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
        # self.displayVar = Tkinter.StringVar()
        # self.displayVar.set("OCR output will be displayed here!")
        self.label3 = Tkinter.Label(self, text="OCR output will be displayed here!")  # text vs textvariable
        self.label3.grid(column=2, row=0)

        # label for displaying candidates
        # self.displayCands = Tkinter.StringVar()
        # self.displayCands.set("Candidates will be displayed here! (to be implemented)")
        self.label4 = Tkinter.Label(self, text="Candidates will be displayed here!")
        self.label4.grid(column=0, row=2)

        # dropdown menu for selecting image processing procedure
        optionList = ["Hybrid", "PrintedBoxesFullSystem", "Adaptive Thresholding", "WholeBoxReprocessSA", "WholeBoxWithKNN", "SARemoveIC", "cv2AdaptiveMean", "cv2AdaptiveGaussian", "AccItemCTest", "AdaptiveParameterExperiment", "HybridItemCTest", "AdaptiveWholeBoxTest", "ReprocessSA", "Advanced", "Basic", "ContourGaussianKernelOtsu", "GaussianKernelAndOtsu", "Otsu"]  # add more later
        self.dropVar = Tkinter.StringVar()
        self.dropVar.set("Hybrid")  # default
        self.dropMenu1 = Tkinter.OptionMenu(self, self.dropVar, *optionList, command=self.func)
        self.dropMenu1.grid(column=1, row=1)

        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)
        self.update()
        self.geometry(self.geometry())

    def OnButtonClickASN(self):
        CSVName = askopenfilename()
        loader = CSVLoader(CSVName)
        self.asn_data = loader.Load()  # what is the data needed for?

        if len(self.asn_data) > 0:
            self.ASN_loaded = True
            self.match_instance = matcher(self.asn_data)

    def OnButtonClickLoad(self):
        filename = askopenfilename()  # prompts user to choose a file
        self.filename = filename

        img2 = Image.open(filename)  # what if user cancelled?
        img2 = img2.resize((self.width, self.height), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img2)

        self.label1.config(image=img2)
        self.label1.image = img2
        processor = img_processor(self.filename)  # need to pass on filename
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
            text, boxes, img = processor.Hybrid()  # and what to do about the iterator?
            img_pic = Image.fromarray(img)
            self.DisplayProcessed(img_pic.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
            interpreter = outputInterpreter()
            text, categories = interpreter.categorizeLines(text)
            # at this point, want to look for candidates, but only if the ASN has already been loaded
            # keep it optional for now to load ASN, in order to faciliate testing
            if self.ASN_loaded:
                categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text, categories)

                # itemcode_cands, itemcode_indeces = self.match_instance.boxFinder(text, categories, img, boxes)
                # print itemcode_cands
                # self.DisplayCands(itemcode_cands)

            index_solid = categories['Solidcode']

            edge_adjustment = 10  # pixels  # introduces issues of out of bound of arrays in rare cases

            coordinates = boxes[index_solid]
            x_top_left = 0
            # x_top_left = max(coordinates[0] - edge_adjustment, 0)
            y_top_left = max(coordinates[1] - edge_adjustment, 0)
            x_bottom_right = img.shape[0]
            # x_bottom_right = min(coordinates[2] + edge_adjustment, img.shape[0])
            y_bottom_right = min(coordinates[3] + edge_adjustment, img.shape[1])

            img = cv2.imread(filename)

            solidcode_arr = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right, :]

            Image.fromarray(solidcode_arr).save('solidslice.png')

        elif self.method in "Adaptive Thresholding":
            text, boxes, img = processor.AdaptiveThresholding()
            img_pic = Image.fromarray(img)
            self.DisplayProcessed(img_pic.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
            interpreter = outputInterpreter()
            text, categories = interpreter.categorizeLines(text)

            if self.ASN_loaded:
                categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text, categories)

                # itemcode_cands, itemcode_indeces = self.match_instance.boxFinder(text, categories, img, boxes)
                # print itemcode_cands
                # self.DisplayCands(itemcode_cands)

                if skip_segmentation:
                    solidassort = text[categories['Solidcode']]
                    print "Solid/assort code: " + str(solidassort)
                    print solidassort_indeces
                    print itemcode_indeces
                    # print which line from ASN it corresponds to.
                    for element in solidassort_indeces:
                        if element in itemcode_indeces:
                            print element
                            print "Found the box"

                else:  # will need to do segmentation
                    print "Itemcode indeces: "
                    print itemcode_indeces

                    index_solid = categories['Solidcode']

                    edge_adjustment = 10  # pixels  # introduces issues of out of bound of arrays in rare cases

                    coordinates = boxes[index_solid]
                    # x_top_left = max(coordinates[0] - edge_adjustment, 0)
                    x_top_left = 0
                    y_top_left = max(coordinates[1] - edge_adjustment, 0)
                    # x_bottom_right = min(coordinates[2] + edge_adjustment, img.shape[0])
                    x_bottom_right = img.shape[0]
                    y_bottom_right = min(coordinates[3] + edge_adjustment, img.shape[1])

                    img = cv2.imread(filename)

                    solidcode_arr = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right, :]

                    Image.fromarray(solidcode_arr).save('solidslice.png')

        elif self.method in "HybridItemCTest":
            if self.ASN_loaded:
                directory = askdirectory()

                count_attempts = 0
                count_correct = 0
                count_correct_not_unique = 0
                errors = []
                not_unique_but_correct = []

                for filename in os.listdir(directory):
                    if filename.endswith(".JPG"):
                        count_attempts += 1
                        # img_read = cv2.imread(os.path.join(directory,filename))
                        processor = img_processor(os.path.join(directory, filename))
                        text, boxes, img = processor.Hybrid()

                        interpreter = outputInterpreter()
                        text, categories = interpreter.categorizeLines(text)
                        categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text,
                                                                                                    categories)  # need to add accuracy check
                        correct_index = self.match_instance.correctFinder(filename)
                        # print itemcode_indeces
                        # print "Correct index: " + str(correct_index)
                        if (correct_index in itemcode_indeces) and unique_ic:
                            count_correct += 1
                        elif (correct_index in itemcode_indeces) and not unique_ic:
                            count_correct_not_unique += 1
                            not_unique_but_correct.append(filename)
                        else:
                            errors.append(filename)
                        print "Categories: "
                        print categories
                        print "Text: "
                        print text
                    print "Images processed: " + str(count_attempts)
                    print "Correctly determined the itemcode: " + str(count_correct)
                print "Line level accuracy solid code: " + str(100.0 * count_correct / count_attempts)
                print "Overall, the number of ICs that were not uniquely determined was: " + str(
                    count_correct_not_unique)
                print "This was applicable to the following images: "
                print not_unique_but_correct
                print "May want to look into the following files: "
                print errors
            else:
                print "Please load ASN before doing accuracy tests m8"

        elif self.method in "WholeBoxWithKNN":
            pick_path = "/Users/sigurdandersberg/PycharmProjects/proj1/Tests/picklewithlist.pickle"
            if self.ASN_loaded:
                directory = askdirectory()
                order = pickle.load(open(pick_path))

                count_attempts = 0
                count_correct_ic = 0
                count_correct_sa = 0
                count_correct_box = 0
                not_unique_but_correct = []
                # errors = []
                count_correct_not_unique = 0
                errors_ic = []
                errors_sa = []

                for filename in order:
                    count_attempts += 1
                    processor = img_processor(os.path.join(directory, filename))
                    text, boxes, img = processor.AdaptiveThresholding()

                    interpreter = outputInterpreter()
                    text, categories = interpreter.categorizeLines(text)
                    categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(
                        text, categories)  # need to add accuracy check
                    correct_index = self.match_instance.correctFinder(filename)

                    index_ic = categories['Itemcode']
                    index_sa = categories['Solidcode']

                    # need to make adjustments, taking blank text lines into account and their missing bounding rectangles.
                    index_sa_adjusted = index_sa
                    sa_adjustment = 0
                    for x in xrange(0, index_sa):
                        line_before = text[x].replace("\n", "")
                        if len(line_before) == 0:
                            sa_adjustment += 1

                    index_sa_adjusted = index_sa - sa_adjustment

                    # find the IC adjustment next:
                    ic_adjustment = 0
                    for x in xrange(0, index_ic):
                        line_before = text[x].replace("\n", "")
                        if len(line_before) == 0:
                            ic_adjustment += 1
                    # index_ic_for_comparison = index_ic - sa_adjustment
                    index_ic_for_comparison = index_ic - ic_adjustment
                    index_sa = index_sa_adjusted  # make sure that text is not being accessed after this point!

                    coordinates = boxes[index_sa]

                    x_top_left = 0
                    y_top_left = coordinates[1]
                    x_bottom_right = img.shape[1]  # assuming x,y are reversed (order) by cv2 image reader.
                    y_bottom_right = coordinates[3] + int(0.01 * img.shape[0])  # too much?
                    # also needed: improved logic for finding the proper index of SA
                    # check: any lines w/ small vertical height near the IC?

                    if type(categories['Itemcode']) is not bool:
                        ic_height = boxes[categories['Itemcode']][3] - boxes[categories['Itemcode']][1]
                        height_threshold = int(0.30 * ic_height)
                    else:
                        height_threshold = 20

                    if (index_sa - index_ic_for_comparison) > 1:
                        if (boxes[index_sa - 1][3] - boxes[index_sa - 1][1]) < height_threshold:
                            y_top_left = boxes[index_sa - 1][1]

                            if (index_sa - index_ic_for_comparison) > 2:
                                if (boxes[index_sa - 2][3] - boxes[index_sa - 2][1]) < height_threshold:
                                    y_top_left = boxes[index_sa - 2][1]
                                    if (index_sa - index_ic_for_comparison) > 3:
                                        if (boxes[index_sa - 3][3] - boxes[index_sa - 3][1]) < height_threshold:
                                            y_top_left = boxes[index_sa - 3][1]
                    elif index_sa - index_ic_for_comparison == 1:  # i.e. directly belwo
                        pass
                    else:
                        print "SA above IC is impossible, something has gone wrong upstream!"  # set to index_ic + 1 as default "best guess"
                        index_sa = index_ic_for_comparison + 1
                        y_top_left = boxes[index_sa][1]
                        y_bottom_right = boxes[index_sa][3]

                    img_snipped = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                    Image.fromarray(img_snipped).save("solidslice.png")

                    # dimension_y, dimension_x = img.shape

                    # background = np.zeros((dimension_y, dimension_x), dtype='float32')
                    # background[y_top_left:y_bottom_right, x_top_left:x_bottom_right] = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                    img_snipped_binary = img[y_top_left:y_bottom_right,
                                         x_top_left:x_bottom_right]  # in numpy array form
                    img_snipped_binary_for_tess = Image.fromarray(img_snipped_binary)
                    repro = reprocessor(img_snipped_binary_for_tess)
                    text_solid_assort = repro.SingleLineTess()  # contains a single guess for the value
                    feed = []
                    feed.append(text_solid_assort)
                    sa_indices, min_dist = self.match_instance.saReprocessed(feed, itemcode_indeces)

                    if min_dist < 0.35:  # need to adjust, and to make consistent along all the checks
                        skip_segmentation = True
                    # check repro against the ASN.

                    # here, update skip_segmentation (boolean), based on the output from the reprocessing of SA line

                    feed = []
                    feed.append(text[categories['Solidcode']])
                    sa_indices_original, dist_original = self.match_instance.saReprocessed(feed, itemcode_indeces)

                    # split into two streams, based on whether segmentation can be skipped or not
                    if not skip_segmentation:
                        solidcands = Segmentation.Fujisawa('solidslice.png')
                        print solidcands

                        sa_indices_from_segmentation, min_dist_from_segmentation = self.match_instance.saReprocessed(
                            solidcands, itemcode_indeces)

                        if (dist_original <= min_dist) and (dist_original <= min_dist_from_segmentation):
                            sa_verdict = sa_indices_original
                        elif (min_dist <= dist_original) and (min_dist <= min_dist_from_segmentation):
                            sa_verdict = sa_indices
                        else:
                            sa_verdict = sa_indices_from_segmentation
                    else:
                        if dist_original <= min_dist:
                            sa_verdict = sa_indices_original
                        else:
                            sa_verdict = sa_indices

                    print itemcode_indeces
                    print sa_verdict

                    # now, want to return the guesses of item code, solid/assort code and the entire box
                    unique_sa = self.match_instance.uniqueSA(itemcode_indeces, sa_verdict)
                    # sa_verdict contains all the indices we are looking for. Use matcher for this
                    if (correct_index in itemcode_indeces) and unique_ic:
                        count_correct_ic += 1
                        print "Found the correct Itemcode"
                        if (correct_index in sa_verdict) and unique_sa:
                            count_correct_box += 1
                            print "Found correct SKU!"

                    if (correct_index in sa_verdict) and unique_sa:
                        count_correct_sa += 1

                    if correct_index not in itemcode_indeces:
                        errors_ic.append(filename)

                    if correct_index not in sa_verdict:
                        errors_sa.append(filename)

                    # accuracies if don't want uniqueness:
                    if (correct_index in itemcode_indeces) and (correct_index in sa_verdict) and (not unique_sa or not unique_ic):
                        count_correct_not_unique += 1
                        not_unique_but_correct.append(filename)

                    print str(filename)

                    self.match_instance.removeDoneBox(correct_index)

                    print "Ended, awaiting next input"
                print "Test completed! "
                print "Correct IC: " + str(count_correct_ic) + " "
                print "Correct SA: " + str(count_correct_sa) + " "
                print "Correct overall, uniquely determined: " + str(count_correct_box) + " "
                print "The corresponding accuracy is: " + str((1.0 * count_correct_box / count_attempts)) + " "
                print "Correct overall but not unique: " + str(count_correct_not_unique) + " "
                print "Errors IC: "
                print errors_ic
                print "Errors SA: "
                print errors_sa
                print "Not unique but correct: "
                print not_unique_but_correct

        elif self.method in "AdaptiveParameterExperiment":
            offset_values = [8, 10, 12]
            block_size_values = [161]
            area_threshold_values = [300, 350, 400, 450, 500]
            if self.ASN_loaded:
                directory = askdirectory() # only ask once

                for bs_val in block_size_values:
                    for os_val in offset_values:
                        for area_val in area_threshold_values:
                            time_begin = time.clock()
                            count_attempts = 0
                            count_correct = 0
                            count_correct_not_unique = 0
                            errors = []
                            not_unique_but_correct = []

                            for filename in os.listdir(directory):
                                if filename.endswith(".JPG"):
                                    count_attempts += 1
                                    # img_read = cv2.imread(os.path.join(directory,filename))
                                    processor = img_processor(os.path.join(directory, filename))
                                    text, boxes, img = processor.AdaptiveThresholdingExperiment(bs_val, os_val, area_val)

                                    interpreter = outputInterpreter()
                                    text, categories = interpreter.categorizeLines(text)
                                    categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text,
                                                                                                                categories)  # need to add accuracy check
                                    correct_index = self.match_instance.correctFinder(filename)
                                    # print itemcode_indeces
                                    # print "Correct index: " + str(correct_index)
                                    if (correct_index in itemcode_indeces) and unique_ic:
                                        count_correct += 1
                                    elif (correct_index in itemcode_indeces) and not unique_ic:
                                        count_correct_not_unique += 1
                                        not_unique_but_correct.append(filename)
                                    else:
                                        errors.append(filename)
                                    print "Categories: "
                                    print categories
                                    print "Text: "
                                    print text
                                print "Images processed: " + str(count_attempts)
                                print "Correctly determined the itemcode: " + str(count_correct)
                            print "Line level accuracy solid code: " + str(100.0 * count_correct / count_attempts)
                            print "Overall, the number of ICs that were not uniquely determined was: " + str(
                                count_correct_not_unique)
                            print "This was applicable to the following images: "
                            print not_unique_but_correct
                            print "May want to look into the following files: "
                            print errors
                            time_end = time.clock()
                            time_taken = time_begin - time_end

                            # collecting the results:
                            with open("output.txt", "a") as outputfile:
                                outputfile.write("Overall correct: " + str(count_correct))
                                outputfile.write("\n")
                                outputfile.write("Overall correct but not unique: " + str(count_correct_not_unique))
                                outputfile.write("\n")
                                outputfile.write("Value for block size " + str(bs_val))
                                outputfile.write("\n")
                                outputfile.write("Value for offset: " + str(os_val))
                                outputfile.write("\n")
                                outputfile.write("Number of images read: " + str(count_attempts))
                                outputfile.write("\n")
                                outputfile.write("Accuracy in %: " + str(100.0*count_correct / count_attempts))
                                outputfile.write("\n")
                                outputfile.write("Time taken: " + str(time_taken))
                                outputfile.write("\n")
                                outputfile.write("Not unique but correct: ")
                                outputfile.writelines(not_unique_but_correct)
                                outputfile.write("\n")
                                outputfile.write("Errors: ")
                                outputfile.writelines(errors)

                            pickle_name = "errors_" + str(bs_val) + "_" + str(os_val) + "_" + str(area_val)

                            with open(os.path.join(os.curdir, pickle_name + ".pickle"), "wb") as f:
                                pickle.dump(errors, f)
                                f.close()

                else:
                    print "Please load ASN before doing accuracy tests m8"

        elif self.method in "AccItemCTest":  # for now, use AdaptiveThresholding for Image processing
            if self.ASN_loaded:
                directory = askdirectory()

                count_attempts = 0
                count_correct = 0
                count_correct_not_unique = 0
                errors = []
                not_unique_but_correct = []

                for filename in os.listdir(directory):
                    if filename.endswith(".JPG"):
                        count_attempts += 1
                        # img_read = cv2.imread(os.path.join(directory,filename))
                        processor = img_processor(os.path.join(directory, filename))
                        text, boxes, img = processor.AdaptiveThresholding()
                        interpreter = outputInterpreter()
                        text, categories = interpreter.categorizeLines(text)
                        #categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text, categories)  # need to add accuracy check
                        categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checkerNoText(text, categories)

                        correct_index = self.match_instance.correctFinder(filename)
                        # print itemcode_indeces
                        # print "Correct index: " + str(correct_index)
                        if (correct_index in itemcode_indeces) and unique_ic:
                            count_correct += 1
                        elif (correct_index in itemcode_indeces) and not unique_ic:
                            count_correct_not_unique += 1
                            not_unique_but_correct.append(filename)
                        else:
                            errors.append(filename)
                        print "Categories: "
                        print categories
                        print "Text: "
                        print text
                    print "Images processed: " + str(count_attempts)
                    print "Correctly determined the itemcode: " + str(count_correct)
                print "Line level accuracy solid code: " + str(100.0 * count_correct / count_attempts)
                print "Overall, the number of ICs that were not uniquely determined was: " + str(count_correct_not_unique)
                print "This was applicable to the following images: "
                print not_unique_but_correct
                print "May want to look into the following files: "
                print errors
            else:
                print "Please load ASN before doing accuracy tests m8"

        elif self.method in "AdaptiveWholeBoxTest":
            if self.ASN_loaded:
                directory = askdirectory()

                count_attempts = 0
                count_correct_ic = 0
                count_correct_sa = 0
                count_correct_box = 0
                not_unique_but_correct = []
                errors = []
                count_correct_not_unique = 0

                for filename in os.listdir(directory):  # note: skip_segmentation's requirement is too strict currently. Try without it at all
                    if filename.endswith(".JPG"):
                        if self.match_instance.inPrintedList(filename):  # then do adaptive thresholding
                            count_attempts += 1
                            processor = img_processor(os.path.join(directory, filename))
                            text, boxes, img = processor.AdaptiveThresholding()

                            interpreter = outputInterpreter()
                            text, categories = interpreter.categorizeLines(text)
                            categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(
                                text, categories)  # need to add accuracy check
                            correct_index = self.match_instance.correctFinder(filename)

                            if (correct_index in itemcode_indeces) and unique_ic:
                                count_correct_ic += 1
                            elif (correct_index in itemcode_indeces) and not unique_ic:
                                count_correct_not_unique += 1
                                not_unique_but_correct.append(filename)
                            else:
                                pass
                                # errors.append(filename)

                            if correct_index in solidassort_indeces:
                                count_correct_sa += 1

                            if (correct_index in itemcode_indeces) and (correct_index in solidassort_indeces):
                                count_correct_box += 1
                                print "Found the box!"
                            else:
                                errors.append(filename)

                            print "Results so far: "
                            print "Overall accuracy" + str(1.0*count_correct_box/count_attempts)
                            print "Tried " + str(count_attempts) + " boxes so far!"

                # print results:
                print "Overall accuracy: " + str(1.0*count_correct_box/count_attempts)
                print "Itemcode accuracy: " + str(1.0*count_correct_ic/count_attempts)
                print "Solidassort accuracy" + str(1.0*count_correct_sa/count_attempts)
                print "Got these wrong:"
                print errors

        elif self.method in "ReprocessSA":
            text, boxes, img = processor.AdaptiveThresholding()
            img_pic = Image.fromarray(img)
            self.DisplayProcessed(img_pic.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
            interpreter = outputInterpreter()
            text, categories = interpreter.categorizeLines(text)

            # analyze the bounding boxes
            # find the mean and sd of the heights of the lines
            heights = []
            for rectangle in boxes:
                heights.append(rectangle[3]-rectangle[1])  # height of the line

            heights = np.asarray(heights)  #  may not be needed
            # mean_height, sd_height = cv2.meanStdDev(heights)  # will not be used yet

            if self.ASN_loaded:
                #categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text, categories)
                categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checkerNoText(text, categories)
                # next, want to extract the S/A code and reprocess it, with a smaller whitelist
                # have boxes from above, and have img still
                correct_index = self.match_instance.correctFinder(self.filename)  # for testing purposes only
                # needd a match instance to find the solidassort indices that fit best.

                index_ic = categories['Itemcode']
                index_sa = categories['Solidcode']

                # need to make adjustments, taking blank text lines into account and their missing bounding rectangles.
                index_sa_adjusted = index_sa
                sa_adjustment = 0
                for x in xrange(0, index_sa):
                    line_before = text[x].replace("\n", "")
                    if len(line_before) == 0:
                        sa_adjustment += 1

                index_sa_adjusted = index_sa - sa_adjustment

                # find the IC adjustment next:
                ic_adjustment = 0
                for x in xrange(0, index_ic):
                    line_before = text[x].replace("\n", "")
                    if len(line_before) == 0:
                        ic_adjustment += 1
                # index_ic_for_comparison = index_ic - sa_adjustment
                index_ic_for_comparison = index_ic - ic_adjustment
                index_sa = index_sa_adjusted  # make sure that text is not being accessed after this point!

                coordinates = boxes[index_sa]

                # to implement: better logic for extracting the solid/assort code
                # based on...vertical thickness of lines
                # position of IC, PCS info, Carton No, text description
                # handle empty line text outputs (check their bounding boxes on line level, are the vertical heights close to 0? )

                x_top_left = 0
                y_top_left = coordinates[1]
                x_bottom_right = img.shape[1]  # assuming x,y are reversed (order) by cv2 image reader.
                y_bottom_right = coordinates[3] + int(0.01*img.shape[0])  # too much?
                # also needed: improved logic for finding the proper index of SA
                # check: any lines w/ small vertical height near the IC?

                if type(categories['Itemcode']) is not bool:
                    ic_height = boxes[categories['Itemcode']][3] - boxes[categories['Itemcode']][1]
                    height_threshold = int(0.30*ic_height)
                else:
                    height_threshold = 20

                if (index_sa - index_ic_for_comparison) > 1:
                    if (boxes[index_sa - 1][3]-boxes[index_sa - 1][1]) < height_threshold:
                        y_top_left = boxes[index_sa-1][1]

                        if (index_sa - index_ic_for_comparison) > 2:
                            if (boxes[index_sa - 2][3] - boxes[index_sa - 2][1]) < height_threshold:
                                y_top_left = boxes[index_sa-2][1]
                                if (index_sa - index_ic_for_comparison) > 3:
                                    if (boxes[index_sa - 3][3] - boxes[index_sa - 3][1]) < height_threshold:
                                        y_top_left = boxes[index_sa-3][1]
                elif index_sa - index_ic_for_comparison == 1:  # i.e. directly belwo
                    pass
                else:
                    print "SA above IC is impossible, something has gone wrong upstream!"  # set to index_ic + 1 as default "best guess"
                    index_sa = index_ic_for_comparison + 1
                    y_top_left = boxes[index_sa][1]
                    y_bottom_right = boxes[index_sa][3]

                img_snipped = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                # img_rgb = cv2.imread(self.filename)
                # img_snipped = img_rgb[y_top_left:y_bottom_right, x_top_left:x_bottom_right]  # region may be too small
                # Image.fromarray(img_snipped).show()  # then pass it on
                Image.fromarray(img_snipped).save("solidslice.png")

                # try: reprocess with Tesseract the extracted Solid/Assort regions
                # for now, assume SOLID
                dimension_y, dimension_x = img.shape
                print dimension_y
                print dimension_x
                # background = np.zeros((dimension_y, dimension_x), dtype='float32')
                # background[y_top_left:y_bottom_right, x_top_left:x_bottom_right] = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                img_snipped_binary = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]  # in numpy array form
                img_snipped_binary_for_tess = Image.fromarray(img_snipped_binary)
                # img_snipped_binary_for_tess.show()
                # Image.fromarray(background).show()
                repro = reprocessor(img_snipped_binary_for_tess)
                text_solid_assort = repro.SingleLineTess()  # contains a single guess for the value
                feed = []
                feed.append(text_solid_assort)
                # should do substrings also, often a redundant digit '1' is added at the front of the output from this stage, while the remaining SC is correct 
                sa_indices, min_dist = self.match_instance.saReprocessed(feed, itemcode_indeces)
                # want the absolute value of the minimum distance also for the purpose of checking whether segmentation is needed or not

                if min_dist < 0.35:
                    skip_segmentation = True
                # check repro against the ASN.

                # here, update skip_segmentation (boolean), based on the output from the reprocessing of SA line

                feed = []
                feed.append(text[categories['Solidcode']])
                sa_indices_original, dist_original = self.match_instance.saReprocessed(feed, itemcode_indeces)

                # split into two streams, based on whether segmentation can be skipped or not
                if not skip_segmentation:
                    solidcands = Segmentation.Fujisawa('solidslice.png')
                    print solidcands

                    sa_indices_from_segmentation, min_dist_from_segmentation = self.match_instance.saReprocessed(solidcands, itemcode_indeces)

                    if (dist_original <= min_dist) and (dist_original <= min_dist_from_segmentation):
                        sa_verdict = sa_indices_original
                    elif (min_dist <= dist_original) and (min_dist <= min_dist_from_segmentation):
                        sa_verdict = sa_indices
                    else:
                        sa_verdict = sa_indices_from_segmentation
                else:
                    if dist_original <= min_dist:
                        sa_verdict = sa_indices_original
                    else:
                        sa_verdict = sa_indices

                print itemcode_indeces
                print sa_verdict
                # compare all the 3 rounds of SA guesses and their min distances to ASN -> form guess about which box it is.

                # now, want to return the guesses of item code, solid/assort code and the entire box


                print str(self.filename)
                print "Ended, awaiting next input"

        elif self.method in "SARemoveIC":
            text, boxes, img = processor.AdaptiveThresholding()
            img_pic = Image.fromarray(img)
            self.DisplayProcessed(img_pic.resize((self.width, self.height), Image.ANTIALIAS))
            self.DisplayOCRText(text)
            interpreter = outputInterpreter()
            text, categories = interpreter.categorizeLines(text)

            # analyze the bounding boxes
            # find the mean and sd of the heights of the lines
            heights = []
            for rectangle in boxes:
                heights.append(rectangle[3] - rectangle[1])  # height of the line

            heights = np.asarray(heights)
            # mean_height, sd_height = cv2.meanStdDev(heights)  # will not be used yet

            if self.ASN_loaded:
                # categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(text, categories)
                categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checkerNoText(
                    text, categories)
                # next, want to extract the S/A code and reprocess it, with a smaller whitelist
                # have boxes from above, and have img still
                index_ic = categories['Itemcode']
                index_sa = categories['Solidcode']

                # need to make adjustments, taking blank text lines into account and their missing bounding rectangles.
                index_sa_adjusted = index_sa
                sa_adjustment = 0
                for x in xrange(0, index_sa):
                    line_before = text[x].replace("\n", "")
                    if len(line_before) == 0:
                        sa_adjustment += 1

                index_sa_adjusted = index_sa - sa_adjustment
                index_ic_for_comparison = index_ic - sa_adjustment

                index_sa = index_sa_adjusted  # make sure that text is not being accessed after this point!

                coordinates = boxes[index_sa]

                # to implement: better logic for extracting the solid/assorrt code
                # based on...vertical thickness of lines
                # position of IC, PCS info, Carton No, text description
                # handle empty line text outputs (check their bounding boxes on line level, are the vertical heights close to 0? )

                x_top_left = 0
                y_top_left = coordinates[1]
                x_bottom_right = img.shape[1]  # assuming x,y are reversed (order) by cv2 image reader.
                y_bottom_right = coordinates[3] + int(0.01 * img.shape[0])  # too much?
                # also needed: improved logic for finding the proper index of SA
                # check: any lines w/ small vertical height near the IC?

                if type(categories['Itemcode']) is not bool:
                    ic_height = boxes[categories['Itemcode']][3] - boxes[categories['Itemcode']][1]
                    height_threshold = int(0.30 * ic_height)
                else:
                    height_threshold = 20

                if (index_sa - index_ic_for_comparison) > 1:
                    if (boxes[index_sa - 1][3] - boxes[index_sa - 1][1]) < height_threshold:
                        y_top_left = boxes[index_sa - 1][1]

                        if (index_sa - index_ic_for_comparison) > 2:
                            if (boxes[index_sa - 2][3] - boxes[index_sa - 2][1]) < height_threshold:
                                y_top_left = boxes[index_sa - 2][1]
                                if (index_sa - index_ic_for_comparison) > 3:
                                    if (boxes[index_sa - 3][3] - boxes[index_sa - 3][1]) < height_threshold:
                                        y_top_left = boxes[index_sa - 3][1]
                elif index_sa - index_ic_for_comparison == 1:  # i.e. directly belwo
                    pass
                else:
                    print "SA above IC is impossible, something has gone wrong upstream!"  # then what to do in this case?
                    # could set to IC + 1 by default?
                    index_sa = index_ic_for_comparison + 1
                    y_top_left = boxes[index_sa][1]
                    y_bottom_right = boxes[index_sa][3]
                    # x-coords should be unaffected

                img_snipped = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right] # black text on white background

                # img_rgb = cv2.imread(self.filename)
                # img_snipped = img_rgb[y_top_left:y_bottom_right, x_top_left:x_bottom_right]  # region may be too small
                Image.fromarray(img_snipped).show()  # then pass it on
                Image.fromarray(img_snipped).save("solidslice.png")

                # try: reprocess with Tesseract the extracted Solid/Assort regions
                # for now, assume SOLID
                dimension_y, dimension_x = img.shape
                print dimension_y
                print dimension_x
                # background = np.zeros((dimension_y, dimension_x), dtype='float32')
                # background[y_top_left:y_bottom_right, x_top_left:x_bottom_right] = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                img_snipped_binary = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]  # in numpy array form
                img_snipped_binary_for_tess = Image.fromarray(img_snipped_binary)
                # img_snipped_binary_for_tess.show()
                # Image.fromarray(background).show()
                # should first remove all traces of the item code from the extracted region before feeding for reprocessing.
                repro = reprocessor(img_snipped_binary_for_tess)
                repro.SingleLineTess()
                print "Ended, awaiting next input"

        elif self.method in "WholeBoxReprocessSA":
            if self.ASN_loaded:
                directory = askdirectory()

                count_attempts = 0
                count_correct_ic = 0
                count_correct_sa = 0
                count_correct_box = 0
                not_unique_but_correct = []
                errors = []
                count_correct_not_unique = 0

                for filename in os.listdir(directory):  # note: skip_segmentation's requirement is too strict currently. Try without it at all
                    if filename.endswith(".JPG"):
                        if self.match_instance.inPrintedList(filename):  # then do adaptive thresholding
                            print str(filename)
                            count_attempts += 1
                            processor = img_processor(os.path.join(directory, filename))
                            text, boxes, img = processor.AdaptiveThresholding()
                            # the following 3 lines are optional - if want to redruce processing time, then cut them out
                            img_pic = Image.fromarray(img)
                            self.DisplayProcessed(img_pic.resize((self.width, self.height), Image.ANTIALIAS))
                            self.DisplayOCRText(text)

                            interpreter = outputInterpreter()
                            text, categories = interpreter.categorizeLines(text)
                            categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checker(
                                text, categories)  # need to add accuracy check
                            correct_index = self.match_instance.correctFinder(filename)

                            # solidassort_line_no = int(categories['Solidcode'])  # integer conversion can cause trouble of there is no index found for S/A line?
                            if type(categories['Solidcode']) is not bool:
                                solidassort_line_no = int(categories['Solidcode'])
                            else:
                                solidassort_line_no = 1 + categories['Itemcode']
                                print "Did not have any OG index for SA line"
                            coordinates = boxes[solidassort_line_no]

                            x_top_left = 0
                            y_top_left = coordinates[1]
                            x_bottom_right = img.shape[1]  # assuming x,y are reversed (order) by cv2 image reader.
                            y_bottom_right = coordinates[3]

                            if (y_bottom_right - y_top_left) > (
                                0.60 * (boxes[int(categories['Itemcode'])][3] - boxes[int(categories['Itemcode'])][1])):
                                pass
                                # img = cv2.imread(self.filename)
                                # img_snipped = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                                # Image.fromarray(img_snipped).show()  # then pass it on
                                # Image.fromarray(img_snipped).save("solidslice.png")
                            else:
                                print "Would not extract S/A code since the bounding rect was not found satisfactory"
                            img_rgb = cv2.imread(self.filename)
                            img_snipped = img_rgb[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                            # Image.fromarray(img_snipped).show()  # then pass it on
                            Image.fromarray(img_snipped).save("solidslice.png")

                            img_snipped_binary = img[y_top_left:y_bottom_right,
                                                                    x_top_left:x_bottom_right]  # in numpy array form
                            img_snipped_binary_for_tess = Image.fromarray(img_snipped_binary)
                            # img_snipped_binary_for_tess.show()
                            repro = reprocessor(img_snipped_binary_for_tess)
                            text_sa_reprocessed = repro.SingleLineTess()  # text

                            # from here, want to find the best candidate for S/A code based on text_sa_reprocessed~
                            sa_indeces_reprocessed = self.match_instance.solidAssortReprocessed(text_sa_reprocessed, itemcode_indeces)
                            solidassort_indeces = sa_indeces_reprocessed  # comment out this line if don't want to use the reprocessed text
                            # at this point, may have a set of ICs (i.e. not uniquely determined)
                            if (correct_index in itemcode_indeces) and unique_ic:
                                count_correct_ic += 1
                            elif (correct_index in itemcode_indeces) and not unique_ic:
                                count_correct_not_unique += 1
                                not_unique_but_correct.append(filename)
                            else:
                                pass
                                # errors.append(filename)

                            if correct_index in solidassort_indeces:
                                count_correct_sa += 1
                            # the below may not be rigorous enough if have several candidates for SKU
                            if (correct_index in itemcode_indeces) and (correct_index in solidassort_indeces):
                                count_correct_box += 1
                                print "Found the box! To see if uniquely determined, need to do further checks! "
                            else:
                                errors.append(filename)

                            print "Results so far: "
                            print "Overall accuracy" + str(1.0*count_correct_box/count_attempts)
                            print "Tried " + str(count_attempts) + " boxes so far!"

                # print results:
                print "Overall accuracy: " + str(1.0*count_correct_box/count_attempts)
                print "Itemcode accuracy: " + str(1.0*count_correct_ic/count_attempts)
                print "Solidassort accuracy" + str(1.0*count_correct_sa/count_attempts)
                print "Got these wrong:"
                print errors

        elif self.method in "PrintedBoxesFullSystem":
            if self.ASN_loaded:
                directory = askdirectory()
                # randomize all the files in the directory, and store the order for future reference.
                name_list = []

                for filename in os.listdir(directory):
                    if filename.endswith(".JPG"):
                        name_list.append(filename)

                shuffle(name_list)
                with open("name_list.pickled", "wb") as f:  # the pickle will get crowded over time if the new lists are written on top of the previous ones
                    pickle.dump(name_list, f)

                count_attempts = 0
                count_correct_ic = 0
                count_correct_sa = 0
                count_correct_box = 0
                not_unique_but_correct = []
                # errors = []
                count_correct_not_unique = 0
                errors_ic = []
                errors_sa = []

                for filename in name_list:
                    if filename.endswith(".JPG"):  # conditions for initiating table removal
                        if self.match_instance.inPrintedList(filename): # don't do if not printed
                            count_attempts += 1
                            processor = img_processor(os.path.join(directory, filename))
                            text, boxes, img = processor.AdaptiveThresholding()
                            interpreter = outputInterpreter()
                            text, categories = interpreter.categorizeLines(text)

                            # Image.fromarray(img).save("binary.png")

                            categories, text, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces = self.match_instance.checkerNoText(
                                text, categories)  # use the checker that does not rely on the text description
                            # next, want to extract the S/A code and reprocess it, with a smaller whitelist
                            # have boxes from above, and have img still
                            correct_index = self.match_instance.correctFinder(filename)  # for testing purposes only. Caution: use filename, not self.filename
                            # need a match instance to find the solidassort indices that fit best.
                            print "Correct index: " + str(correct_index)

                            feed = []
                            feed.append(text[categories['Solidcode']])
                            sa_indices_original, dist_original = self.match_instance.saReprocessed(feed,
                                                                                                   itemcode_indeces)
                            # consider: check again if want to do any segmentation
                            # prior to extracting the SA code, would like to

                            rp_table = reprocessor(Image.fromarray(img))

                            result = rp_table.removeTables(img)  # want to remove the table (largest rectangular contour)

                            if type(result) is not bool:
                                img_rt = result
                                Image.fromarray(img_rt).save("binarynotable.png")
                                proc2 = img_processor("binarynotable.png")
                                text_rt, boxes_rt, img_rt = proc2.AdaptiveThresholding()

                                text_rt, categories = interpreter.categorizeLines(text_rt)

                                cat_rt, text_rt, ic_ind_rt, unique_ic_rt, skip_seg_rt, sa_ind_rt = self.match_instance.checkerNoText(text_rt, categories)
                                # then pass on to matcher instance

                                feed = []
                                feed.append(text_rt[cat_rt['Solidcode']])  # because it must be in form of a list
                                sa_indices_rt, dist_rt = self.match_instance.saReprocessed(feed, ic_ind_rt)
                                # compare the results from this stage to those of the previous method (no removal of table)
                                if dist_rt < dist_original:
                                    sa_indices_original = sa_indices_rt  # then move on from here
                                    itemcode_indeces = ic_ind_rt
                                    categories = cat_rt
                                    text = text_rt
                                    boxes = boxes_rt
                                    solidassort_indeces = sa_ind_rt
                                    skip_segmentation = skip_seg_rt

                            index_ic = categories['Itemcode']
                            index_sa = categories['Solidcode']

                            # need to make adjustments, taking blank text lines into account and their missing bounding rectangles.
                            sa_adjustment = 0
                            for x in xrange(0, index_sa):
                                line_before = text[x].replace("\n", "")
                                if len(line_before) == 0:
                                    sa_adjustment += 1

                            index_sa_adjusted = index_sa - sa_adjustment

                            # find the IC adjustment next:
                            ic_adjustment = 0
                            for x in xrange(0, index_ic):
                                line_before = text[x].replace("\n", "")
                                if len(line_before) == 0:
                                    ic_adjustment += 1
                            # index_ic_for_comparison = index_ic - sa_adjustment
                            index_ic_for_comparison = index_ic - ic_adjustment
                            index_sa = index_sa_adjusted  # make sure that text is not being accessed after this point!

                            coordinates = boxes[index_sa]

                            # to implement: better logic for extracting the solid/assort code
                            # based on...vertical thickness of lines
                            # position of IC, PCS info, Carton No, text description
                            # handle empty line text outputs (check their bounding boxes on line level, are the vertical heights close to 0? )

                            x_top_left = 0
                            y_top_left = coordinates[1]
                            x_bottom_right = img.shape[1]  # assuming x,y are reversed (order) by cv2 image reader.
                            y_bottom_right = coordinates[3] + int(0.01 * img.shape[0])  # too much?
                            # also needed: improved logic for finding the proper index of SA
                            # check: any lines w/ small vertical height near the IC?

                            if type(categories['Itemcode']) is not bool:
                                ic_height = boxes[categories['Itemcode']][3] - boxes[categories['Itemcode']][1]
                                height_threshold = int(0.30 * ic_height)
                            else:
                                height_threshold = 20

                            if (index_sa - index_ic_for_comparison) > 1:
                                if (boxes[index_sa - 1][3] - boxes[index_sa - 1][1]) < height_threshold:
                                    y_top_left = boxes[index_sa - 1][1]

                                    if (index_sa - index_ic_for_comparison) > 2:
                                        if (boxes[index_sa - 2][3] - boxes[index_sa - 2][1]) < height_threshold:
                                            y_top_left = boxes[index_sa - 2][1]
                                            if (index_sa - index_ic_for_comparison) > 3:
                                                if (boxes[index_sa - 3][3] - boxes[index_sa - 3][
                                                    1]) < height_threshold:
                                                    y_top_left = boxes[index_sa - 3][1]
                            elif index_sa - index_ic_for_comparison == 1:  # i.e. directly below
                                pass
                            else:
                                print "SA above IC is impossible, something has gone wrong upstream!"  # set to index_ic + 1 as default "best guess
                                index_sa = index_ic_for_comparison + 1
                                try:
                                    y_top_left = boxes[index_sa][1]
                                    y_bottom_right = boxes[index_sa][3]
                                except IndexError:
                                    y_top_left = boxes[index_sa-1][1]
                                    y_bottom_right = boxes[index_sa-1][3]

                            img_snipped = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                            # img_rgb = cv2.imread(self.filename)
                            # img_snipped = img_rgb[y_top_left:y_bottom_right, x_top_left:x_bottom_right]  # region may be too small
                            # Image.fromarray(img_snipped).show()  # then pass it on
                            Image.fromarray(img_snipped).save("solidslice.png")

                            # try: reprocess with Tesseract the extracted Solid/Assort regions
                            # for now, assume SOLID
                            dimension_y, dimension_x = img.shape
                            print dimension_y
                            print dimension_x
                            # background = np.zeros((dimension_y, dimension_x), dtype='float32')
                            # background[y_top_left:y_bottom_right, x_top_left:x_bottom_right] = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                            img_snipped_binary = img[y_top_left:y_bottom_right,
                                                 x_top_left:x_bottom_right]  # in numpy array form
                            img_snipped_binary_for_tess = Image.fromarray(img_snipped_binary)

                            repro = reprocessor(img_snipped_binary_for_tess)
                            text_solid_assort = repro.SingleLineTess()  # contains a single guess for the value
                            feed = []
                            feed.append(text_solid_assort)
                            # should do substrings also, often a redundant digit '1' is added at the front of the output from this stage, while the remaining SC is correct
                            sa_indices, min_dist = self.match_instance.saReprocessed(text_solid_assort,
                                                                                     itemcode_indeces)
                            # want the absolute value of the minimum distance also for the purpose of checking whether segmentation is needed or not

                            if min_dist < 0.25:  #changed from 0.35
                                skip_segmentation = True

                            # check again here if want to do segmentation or not

                            # need more appropriate comparison of the results for SA code from both first and second passes
                            # dist_original
                            # min_dist_reprocess
                            min_dist_reprocess = min_dist  # corresponds to sa_indices
                            sa_indices_reprocess = sa_indices
                            min_dist_first = dist_original  # corresponds to sa_indices_original
                            sa_indices_first = solidassort_indeces  # check

                            if min_dist_first < 0.25:  # changed from 0.35
                                skip_segmentation = True

                            # split into two streams, based on whether segmentation can be skipped or not
                            if not skip_segmentation:
                                solidcands = Segmentation.Fujisawa('solidslice.png')
                                print solidcands

                                sa_indices_from_segmentation, min_dist_from_segmentation = self.match_instance.saReprocessed(
                                    solidcands, itemcode_indeces)

                                if (dist_original <= min_dist) and (dist_original <= min_dist_from_segmentation):
                                    sa_verdict = sa_indices_original
                                elif (min_dist <= dist_original) and (min_dist <= min_dist_from_segmentation):
                                    sa_verdict = sa_indices
                                else:
                                    sa_verdict = sa_indices_from_segmentation
                            else:
                                if min_dist_reprocess < min_dist_first:
                                    sa_verdict = sa_indices_reprocess
                                else:
                                    sa_verdict = sa_indices_first

                            print itemcode_indeces
                            print sa_verdict
                            # compare all the 3 rounds of SA guesses and their min distances to ASN -> form guess about which box it is.

                            # now, want to return the guesses of item code, solid/assort code and the entire box

                            print str(filename)
                            print "Starting next input"

                            # now, want to return the guesses of item code, solid/assort code and the entire box
                            unique_sa = self.match_instance.uniqueSA(itemcode_indeces, sa_verdict)
                            # sa_verdict contains all the indices we are looking for. Use matcher for this
                            """if (correct_index in itemcode_indeces) and unique_ic:
                                count_correct_ic += 1
                                print "Found the correct Itemcode"
                                if (correct_index in sa_verdict) and unique_sa:
                                    count_correct_box += 1
                                    print "Found correct SKU!"""
                            if correct_index in itemcode_indeces:
                                count_correct_ic += 1
                                print "Found correct itemcode!"
                                if correct_index in sa_verdict:
                                    count_correct_box += 1
                                    print "Found correct SKU!"

                            if correct_index in sa_verdict:
                                count_correct_sa += 1
                                print "Found correct SA!"

                            if correct_index not in itemcode_indeces:
                                errors_ic.append(filename)

                            if correct_index not in sa_verdict:
                                errors_sa.append(filename)

                            # accuracies if don't want uniqueness:
                            if (correct_index in itemcode_indeces) and (correct_index in sa_verdict) and (
                                not unique_sa or not unique_ic):
                                count_correct_not_unique += 1
                                not_unique_but_correct.append(filename)

                            print str(filename)

                            self.match_instance.removeDoneBox(correct_index)

                            print "So far, have done " + str(count_attempts) + " boxes, and the accuracy thus far is: " + str((1.0 * count_correct_box / count_attempts))

                            print "Ended, awaiting next input"
                        else:
                            correct_index = self.match_instance.correctFinder(filename)
                            self.match_instance.removeDoneBox(correct_index)
                print "Test completed! "
                print "Correct IC: " + str(count_correct_ic) + " "
                print "Correct SA: " + str(count_correct_sa) + " "
                print "Correct overall, uniquely determined: " + str(count_correct_box) + " "
                print "The corresponding accuracy is: " + str((1.0 * count_correct_box / count_attempts)) + " "
                print "Correct overall but not unique: " + str(count_correct_not_unique) + " "
                print "Errors IC: "
                print errors_ic
                print "Errors SA: "
                print errors_sa
                print "Not unique but correct: "
                print not_unique_but_correct
                print "Number of boxes tested: " + str(count_attempts)


        elif self.method in "cv2AdaptiveMean":  # needs to be fixed
            processor.cv2AdaptiveMean()

        elif self.method in "cv2AdaptiveGaussian":  # need fix
            processor.cv2AdaptiveGaussian()
        else:
            print "Method selection error!"

    def DisplayProcessed(self, image_processed):
        image_processed = ImageTk.PhotoImage(image_processed)
        self.label2.config(image=image_processed)
        self.label2.image = image_processed

    def func(self, value):  # what does this do? is it called at all?
        self.method = value
        # print self.method

    def DisplayOCRText(self, text):
        if len(text) > 400:
            text = "Use other image processing methods!"
        self.label3.config(text=text)

    def DisplayCands(self, candidates):  # set area for the candidates, should have one cand. per line, and at most 10 cands
        if len(candidates) == 0:
            candidates = "No candidates!"
        self.label4.config(text=candidates)


class reprocessor():
    def __init__(self, img):
        self.img = img # takes a PIL image as input

    def SingleLineTess(self):
        with PyTessBaseAPI(psm=7) as api: # 7 may also work. Try 8 also?
            api.SetVariable("tessedit_char_whitelist", "-0123456789IFRL. ")  # ABCDEFGHIJKLMNOPQRSTUVWXYZ
            api.SetImage(self.img)
            text = api.GetUTF8Text()
            print text
            text = text.encode("utf-8")  # if want to store it in a txt file
            print api.AllWordConfidences()

        return text

    def removeTables(self, image):
        img = image
        abac, contours, hiearchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        area_threshold = 90000  # even 6000 may be too small of a threshold
        # large_contours = []
        area_img = img.shape[0] * img.shape[1]
        area_upper_bound = math.floor(0.85 * area_img)

        largest_cont = 0
        largest_area = 0

        for cont in contours:
            if (cv2.contourArea(cont) > area_threshold) and (cv2.contourArea(cont) < area_upper_bound):
                rect = cv2.boundingRect(cont)
                area_rect = math.floor(rect[2] * rect[3])
                ratio_areas = 1.0 * cv2.contourArea(cont) / area_rect
                print ratio_areas
                if ratio_areas > 0.80:
                    # assume it's a table of label in this case
                    # large_contours.append(cont)
                    # rect = cv2.minAreaRect(cont)  # normally, the rotation would be minimal
                    # print rect
                    # box = cv2.boxPoints(rect)  # need to get area of the bounding box
                    # print box
                    # large_contours.append(cont)
                    # box = np.int0(box)
                    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                    if cv2.contourArea(cont) > largest_area:
                        largest_area = cv2.contourArea(cont)
                        largest_cont = cont

        if type(largest_cont) is not int:
            print len(largest_cont)  # need to verify that length is 1
            rect = cv2.boundingRect(largest_cont)

            black_bg = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')  # in terms of y, x
            black_bg[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[
                2]] = 255  # all pixels within the outer bounding rectangle are supposed to turn white

            # up until now, img is with very light color, on black background?
            # want to have black text on white background, want
            img = 255 * img
            result = img + black_bg

            return result
        else:
            return False


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

    def removeTables(self):
        img = cv2.imread(self.filename)  # should be binary image (or more exact, the output directly from the adaptive thresh)
        ab, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        area_threshold = 90000  # even 6000 may be too small of a threshold
        large_contours = []
        area_img = img.shape[0] * img.shape[1]
        area_upper_bound = math.floor(0.85 * area_img)

        largest_cont = 0
        largest_area = 0

        for cont in contours:
            if (cv2.contourArea(cont) > area_threshold) and (cv2.contourArea(cont) < area_upper_bound):
                rect = cv2.boundingRect(cont)
                area_rect = math.floor(rect[2] * rect[3])
                ratio_areas = 1.0 * cv2.contourArea(cont) / area_rect
                print ratio_areas
                if ratio_areas > 0.80:
                    # assume it's a table of label in this case
                    large_contours.append(cont)
                    # rect = cv2.minAreaRect(cont)  # normally, the rotation would be minimal
                    # print rect
                    # box = cv2.boxPoints(rect)  # need to get area of the bounding box
                    # print box
                    # large_contours.append(cont)
                    # box = np.int0(box)
                    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                    if cv2.contourArea(cont) > largest_area:
                        largest_area = cv2.contourArea(cont)
                        largest_cont = cont

        if type(largest_cont) is not int:
            print len(largest_cont)  # need to verify that length is 1
            rect = cv2.boundingRect(largest_cont)

            black_bg = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')  # in terms of y, x
            black_bg[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[
                2]] = 255  # all pixels within the outer bounding rectangle are supposed to turn white

            # up until now, img is with very light color, on black background?
            # want to have black text on white background, want
            img = 255 * img
            result = img + black_bg

            # may want to call this part form outside the class, in order to avoid instances within instances
            Image.fromarray(result).save("aftertablegone.png")
            fn = "aftertablegone.png"
            text, boxes, img = self.AdaptiveThresholding()

            return text, boxes, img
        else:
            return False, False, False

    def Basic(self, thresholded, img_array):  # for the purpose of debugging, will also draw the bounding rectangles for where tesseract thinks there are characters
        load_time = time.clock()
        with PyTessBaseAPI(psm=6) as api: # originally psm 6. Tried psm=4 without too good results.
            api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
            api.SetImage(thresholded)
            text_output = api.GetUTF8Text().encode('utf-8')
            print text_output  # can remove
            # image_processed = api.GetThresholdedImage()
            end_time = time.clock()
            print "Tesseract time: " + str(end_time - load_time)
            iterator = api.GetIterator()
            iterator.Begin()
            level = RIL.TEXTLINE
            boxes = []
            for r in iterate_level(iterator, level):
                boxes.append(r.BoundingBox(level))
            print boxes
            img_from_tess = img_array.copy()
            # img_from_tess = img_array  # use this line if want to have the bounding rectangles
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
            # also return images

    def cv2AdaptiveMean(self):  # may want to apply a blur first?
        area_threshold = 350
        img = cv2.imread(self.filename)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(img_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 161, 10)  # very slow?
        a, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        large_cnt = []

        for cnt in contours:
            if cv2.contourArea(cnt) > area_threshold:
                large_cnt.append(cnt)

        black_bg = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        cv2.drawContours(black_bg, large_cnt, -1, color=255, thickness=1)
        Image.fromarray(black_bg).show()
        print "cv2AdaptiveMean done!"

        # may want to return something

    def cv2AdaptiveGaussian(self):
        area_threshold = 350
        img = cv2.imread(self.filename)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(img_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 161, 10)
        a, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        large_cnt = []

        for cnt in contours:
            if cv2.contourArea(cnt) > area_threshold:
                large_cnt.append(cnt)

        black_bg = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        cv2.drawContours(black_bg, large_cnt, -1, color=255, thickness=1)
        Image.fromarray(black_bg).show()

        print "cv2AdaptiveGaussian done!"

    def Hybrid(self):
        tic = time.clock()
        img = cv2.imread(self.filename)
        load_time = time.clock()
        print "Load time: " + str(load_time - tic)
        contour_area_min = 350  # 700 to 800 seem to work well. Was 400

        # next split image into the three RGB channels
        img_red = img[:, :, 0]
        img_green = img[:, :, 1]
        img_blue = img[:, :, 2]
        print img_red.shape
        # perform Canny edge detector on each channel since text may be found in any of the channels -- but which parameters to use?
        # edge_red = feature.canny(img_red)
        # edge_green = feature.canny(img_green)
        # edge_blue = feature.canny(img_blue)
        edge_red = cv2.Canny(img_red, 50, 100)  # originally used thresholds 50, 100 instead of 100, 200
        edge_green = cv2.Canny(img_green, 50, 100)
        edge_blue = cv2.Canny(img_blue, 50, 100)
        # Can try cv2.Canny(image_input, lower, upper) with e.g. lower=100, upper=200
        # the params are for removing and detecting noise
        edge_assimilated = np.logical_or(edge_red, np.logical_or(edge_green, edge_blue))  # boolean array
        canny_time = time.clock()
        print "Canny time: " + str(canny_time - load_time)
        # Next, want to do both horizontal and vertical dilation with 1x3 and 3x1 structuring elements
        # Note: paper suggests 1x3 and 3x1, but in our application 5x1 and 1x5 might work better
        strel1 = np.zeros((5, 5))
        for i in range(0, 5):
            strel1[2][i] = 1
        # horizontal dilation
        edge_dim_1 = ndi.binary_dilation(edge_assimilated, structure=strel1)

        strel2 = np.zeros((5, 5))

        for j in range(0, 5):
            strel2[j][2] = 1
        # vertical dilation
        edge_dim_2 = ndi.binary_dilation(edge_assimilated, structure=strel2)
        # combine the results with OR operator
        edges_dil_comb = np.logical_or(edge_dim_1, edge_dim_2)
        # all_labels = measure.label(edges_dil_comb, neighbors=8, connectivity=2).astype('uint8')  # same dimensions as image
        all_labels = edges_dil_comb.astype('uint8')
        # the above step may not be necessary? If skip, convert edges_dil_comb to 'uint8'
        im2, abac, hierarchy = cv2.findContours(all_labels.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        hierarchy = hierarchy[0]  # get the useful stuff only

        # filter contours based on area (reject small ones, which are definitely noise)
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
        """for i in range(0, len(accepted_conts)):
            # bg_for_density = np.zeros((img_red.shape[0], img_red.shape[1]), dtype='uint8')
            # cv2.drawContours(bg_for_density, accepted_conts, i, color=255, thickness=-1)
            # pts_col = np.where(bg_for_density == 255)
            cv2.drawContours(bg_for_density_prime, accepted_conts, i, color=255, thickness=-1)
            # Get values, both zero and non-zero
            # vals_for_numerator = bg_for_final[pts_col[0], pts_col[1]]
            # vals_for_denominator = bg_for_density[pts_col[0], pts_col[1]]

            # numerator = 0

            # for ele in vals_for_numerator:
                # if ele > 100:
                    # numerator += 1

            # denominator = len(vals_for_denominator)
            # print denominator

            # density = 1.0 * numerator / denominator
            # print " Density: " + str(density)
            # threshold_density = 0.20  # calibration needed
            # The following part does pretty much nothing so far
            # if density > threshold_density:  # this rarely triggers, look into how to apply it
                # bg_for_final[pts_col[0], pts_col[1]] = 0
                # print " Triggered, m8! "
"""
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
        text_ret, boxes = self.Basic(for_tess_img, for_tess_arr)

        return text_ret, boxes, img_ret

    def AdaptiveThresholdingExperiment(self, block_size, offset, area_lower_bound):
        img_read = cv2.imread(self.filename)

        grayscale = cv2.cvtColor(img_read,
                                 cv2.COLOR_BGR2GRAY)  # potential improvement: using multiple color channels and combining results

        block_size = block_size
        offset = offset
        area_lower_bound = area_lower_bound
        binar_adaptive = threshold_adaptive(grayscale, block_size, offset=offset)

        # next, do noise removal
        noisy = binar_adaptive.astype('uint8') * 255

        im2, contours, hierarchy = cv2.findContours(noisy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        large_contours = []

        for cnt in contours:
            if cv2.contourArea(cnt) > area_lower_bound:
                large_contours.append(cnt)

        black_bg = np.zeros((img_read.shape[0], img_read.shape[1]), dtype='uint8')
        cv2.drawContours(black_bg, large_contours, -1, color=255, thickness=-1)
        # Image.fromarray(black_bg).show()  # black text on white background
        combined = np.logical_and(255 - black_bg, 255 - noisy)  # why are some tiny pixels left here?
        combined = combined.astype('uint8')*255

        img_for_tess = Image.fromarray(combined)

        text_output, boxes = self.Basic(img_for_tess, combined)

        return text_output, boxes, (255-combined)

    def AdaptiveThresholding(self):
        img_read = cv2.imread(self.filename)
        area_lower_bound = 350  # originally 300..and 200. Now changed to 400 (17 Mar 2017). 350 from 24 Mar

        grayscale = cv2.cvtColor(img_read,
                                 cv2.COLOR_BGR2GRAY)  # potential improvement: using multiple color channels and combining results

        block_size = 161  # originally 121, 161 from 17 Mar 2017
        binar_adaptive = threshold_adaptive(grayscale, block_size, offset=10)  # offset originally 24, changed to 10 on 17 Mar 2017

        # next, do noise removal
        noisy = binar_adaptive.astype('uint8') * 255

        Image.fromarray(noisy).save("noisy_binary.png")  # can be commented out

        im2, contours, hierarchy = cv2.findContours(noisy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        large_contours = []

        for cnt in contours:
            if cv2.contourArea(cnt) > area_lower_bound:
                large_contours.append(cnt)

        black_bg = np.zeros((img_read.shape[0], img_read.shape[1]), dtype='uint8')
        cv2.drawContours(black_bg, large_contours, -1, color=255, thickness=-1)
        # Image.fromarray(black_bg).show()  # black text on white background
        combined = np.logical_and(255 - black_bg, 255 - noisy)  # why are some tiny pixels left here?
        combined = combined.astype('uint8')*255

        img_for_tess = Image.fromarray(combined)

        text_output, boxes = self.Basic(img_for_tess, combined)

        return text_output, boxes, (255-combined)

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
            # api.GetThresholdedImage().show()  # optional, for debugging

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
        # img_for_tess.show()

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
        # img_for_tess.show()

        # Tesseract part
        with PyTessBaseAPI(psm=6) as api:
            api.SetImage(img_for_tess)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()
            print text_output  # to be removed
        return text_output, image_processed

    def Otsu(self):
        """Uses only Otsu's binarization, followed by Tesseract for OCR. May want to do an updated version with noise removal"""
        # load image
        img_raw = cv2.imread(self.filename)
        img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)[1]
        img_for_tess = Image.fromarray(img_binary)
        img_for_tess.save('otsu.png')
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

class CSVLoader():  # also functions as CSV handler
    """Loads the ASN file from csv format, and stores all the values in an array"""
    def __init__(self, ASNfile):
        self.ASNfile = ASNfile
        self.ASN_data = []

    def Load(self):
        """Load CSV, split the data based on that they are"""
        asn_data = []

        with open(self.ASNfile, "rb") as csvfile:  # load the ASN from csv file, store in array (line by line)
            spamreader = csv.reader(csvfile, delimiter=';',
                                    quotechar='|')  # Spamreader is an object..need to save the info somehow

            for row in spamreader:  # Row only contains 1 row at any given time
                print ', '.join(row)
                asn_data.append(row)
        csvfile.close()

        # want to remove empty lines from the list of ASN elements (go bottom-up)
        nonempty_found = False
        while not nonempty_found:
            element = asn_data[len(asn_data)-1]
            if (len(element[0]) > 1) or (element[1] > 1) or (element[2] > 1) or (element[3] > 1):
                nonempty_found = True
            else:
                asn_data.pop()
        # self.ASN_data = asn_data
        # ata_arr = np.asarray(asn_data)

        return asn_data


def INSERTION(A, cost=1):
  return cost


def DELETION(A, cost=1):
  return cost

def DELETIONNOCOST(A, cost=0):
    return cost

def SUBSTITUTION(A, B, cost=0.15):  # want custom weights. Confusion matrix please!
  if (A in "O" or "o") and (B in "0"):
          return 0.04
  elif (A in "I" or "i" or "L" or "l") and (B in "1"):
      return 0.05
  elif (A in "B" or "b") and (B in "8"):
      return 0.055
  elif (A in ".") and (B in "-"):
      return 0.08
  else:
    return cost

def DELETIONSMATCHER(A, cost=1): # Does NOT need to be integer
    if A in " ":
        return 0
    elif A.isalpha():
        return 0.04
    elif A.isdigit():
        return 0.04  # was originally 0.2
    else:
        return cost


Trace = collections.namedtuple("Trace", ["cost", "ops"])


class WagnerFischer(object):

    """
    An object representing a (set of) Levenshtein alignments between two
    iterable objects (they need not be strings). The cost of the optimal
    alignment is scored in `self.cost`, and all Levenshtein alignments can
    be generated using self.alignments()`.

    Basic tests:

    >>> WagnerFischer("god", "gawd").cost
    2
    >>> WagnerFischer("sitting", "kitten").cost
    3
    >>> WagnerFischer("bana", "banananana").cost
    6
    >>> WagnerFischer("bana", "bana").cost
    0
    >>> WagnerFischer("banana", "angioplastical").cost
    11
    >>> WagnerFischer("angioplastical", "banana").cost
    11
    >>> WagnerFischer("Saturday", "Sunday").cost
    3

    IDS tests:

    >>> WagnerFischer("doytauvab", "doyvautab").IDS() == {"S": 2.0}
    True
    >>> WagnerFischer("kitten", "sitting").IDS() == {"I": 1.0, "S": 2.0}
    True

    Detect insertion vs. deletion:

    >>> thesmalldog = "the small dog".split()
    >>> thebigdog = "the big dog".split()
    >>> bigdog = "big dog".split()
    >>> sub_inf = lambda A, B: float("inf")

    # Deletion.
    >>> wf = WagnerFischer(thebigdog, bigdog, substitution=sub_inf)
    >>> wf.IDS() == {"D": 1.0}
    True

    # Insertion.
    >>> wf = WagnerFischer(bigdog, thebigdog, substitution=sub_inf)
    >>> wf.IDS() == {"I": 1.0}
    True

    # Neither.
    >>> wf = WagnerFischer(thebigdog, thesmalldog, substitution=sub_inf)
    >>> wf.IDS() == {"I": 1.0, "D": 1.0}
    True
    """

    # Initializes pretty printer (shared across all class instances).
    pprinter = pprint.PrettyPrinter(width=75)

    def __init__(self, A, B, insertion=INSERTION, deletion=DELETION,
                 substitution=SUBSTITUTION):
        # Stores cost functions in a dictionary for programmatic access.
        self.costs = {"I": insertion, "D": deletion, "S": substitution}
        # Initializes table.
        self.asz = len(A)
        self.bsz = len(B)
        self._table = [[None for _ in range(self.bsz + 1)] for
                       _ in range(self.asz + 1)]
        # From now on, all indexing done using self.__getitem__.
        ## Fills in edges.
        self[0][0] = Trace(0, {"O"})  # Start cell.
        for i in range(1, self.asz + 1):
            self[i][0] = Trace(self[i - 1][0].cost + self.costs["D"](A[i - 1]),
                               {"D"})
        for j in range(1, self.bsz + 1):
            self[0][j] = Trace(self[0][j - 1].cost + self.costs["I"](B[j - 1]),
                               {"I"})
        ## Fills in rest.
        for i in range(len(A)):
            for j in range(len(B)):
                # Cleans it up in case there are more than one check for match
                # first, as it is always the cheapest option.
                if A[i] == B[j]:
                    self[i + 1][j + 1] = Trace(self[i][j].cost, {"M"})
                # Checks for other types.
                else:
                    costD = self[i][j + 1].cost + self.costs["D"](A[i])
                    costI = self[i + 1][j].cost + self.costs["I"](B[j])
                    costS = self[i][j].cost + self.costs["S"](A[i], B[j])
                    min_val = min(costI, costD, costS)
                    trace = Trace(min_val, set())
                    # Adds _all_ operations matching minimum value.
                    if costD == min_val:
                        trace.ops.add("D")
                    if costI == min_val:
                        trace.ops.add("I")
                    if costS == min_val:
                        trace.ops.add("S")
                    self[i + 1][j + 1] = trace
        # Stores optimum cost as a property.
        self.cost = self[-1][-1].cost

    def __repr__(self):
        return self.pprinter.pformat(self._table)

    def __iter__(self):
        for row in self._table:
            yield row

    def __getitem__(self, i):
        """
        Returns the i-th row of the table, which is a list and so
        can be indexed. Therefore, e.g.,  self[2][3] == self._table[2][3]
        """
        return self._table[i]

    # Stuff for generating alignments.

    def _stepback(self, i, j, trace, path_back):
        """
        Given a cell location (i, j) and a Trace object trace, generate
        all traces they point back to in the table
        """
        for op in trace.ops:
            if op == "M":
                yield i - 1, j - 1, self[i - 1][j - 1], path_back + ["M"]
            elif op == "I":
                yield i, j - 1, self[i][j - 1], path_back + ["I"]
            elif op == "D":
                yield i - 1, j, self[i - 1][j], path_back + ["D"]
            elif op == "S":
                yield i - 1, j - 1, self[i - 1][j - 1], path_back + ["S"]
            elif op == "O":
                return  # Origin cell, so we"re done.
            else:
                raise ValueError("Unknown op {!r}".format(op))

    def alignments(self):
        """
        Generate all alignments with optimal-cost via breadth-first
        traversal of the graph of all optimal-cost (reverse) paths
        implicit in the dynamic programming table
        """
        # Each cell of the queue is a tuple of (i, j, trace, path_back)
        # where i, j is the current index, trace is the trace object at
        # this cell, and path_back is a reversed list of edit operations
        # which is initialized as an empty list.
        queue = collections.deque(self._stepback(self.asz, self.bsz,
                                                 self[-1][-1], []))
        while queue:
            (i, j, trace, path_back) = queue.popleft()
            if trace.ops == {"O"}:
                # We have reached the origin, the end of a reverse path, so
                # yield the list of edit operations in reverse.
                yield path_back[::-1]
                continue
            queue.extend(self._stepback(i, j, trace, path_back))

    def IDS(self):
        """
        Estimates insertions, deletions, and substitution _count_ (not
        costs). Non-integer values arise when there are multiple possible
        alignments with the same cost.
        """
        npaths = 0
        opcounts = collections.Counter()
        for alignment in self.alignments():
            # Counts edit types for this path, ignoring "M" (which is free).
            opcounts += collections.Counter(op for op in alignment if op != "M")
            npaths += 1
        # Averages over all paths.
        return collections.Counter({o: c / npaths for (o, c) in
                                    opcounts.items()})

class matcher():
    def __init__(self, ASN_data):
        self.asn = ASN_data

    def inPrintedList(self, filename):
        """Returns true if the parcel has only printed text, false if there is any handwriting in IC or S/A"""
        asn = self.asn

        ret_val = False

        for index in range(1, len(asn)):
            if str(asn[index][3]) in str(filename):
                if str(asn[index][4]) in "Print":
                    ret_val = True
                    break

        return ret_val

    def removeDoneBox(self, index):
        self.asn.pop(int(index))  # should not be necessary to convert to int
        # check if this pops the entire asn away?

    def uniqueSA(self, itemcode_indices, sa_indices):
        asn = self.asn  # structure: Text, IC, SA, Filename, HW/printed

        sa_indices = sa_indices
        itemcode_indices = itemcode_indices

        seen_sa_guesses = []

        for sa in sa_indices:
            if sa in itemcode_indices:  # otherwise, do not consider
                if str(asn[sa][2]) not in seen_sa_guesses:
                    seen_sa_guesses.append(str(asn[sa][2]))

        if len(seen_sa_guesses) == 1:
            return True
        else:
            return False

    def correctFinder(self, filename):
        # img name is on form IMG_DDDD.JPG. DDDD should be found in self.asn[index][3]
        asn = self.asn

        yo = re.findall(r'\d{4}', filename, re.UNICODE)  # look for the 4 digits in a row in filename
        if len(yo) == 1:
            code = yo[0]

            for k in range(1, len(asn)):
                if code in asn[k][3]:
                    return k

        else:
            print "Error in handling filename"
            return False
    # should replace the below with the one found in categorizelinestest.py
    def itemcodeFinderOutdated(self, text_lines, categories):
        """Find the best match(es) for item code. Return the corresponding candidates
        Should be based on both text and item code line"
        Should only run if ASN is loaded. Can later be used to combine results from both images of the box (2 sides) to be more forgiving while remaining robust"""

        asn_data = self.asn_data  # from the creation of the object

        # categories is a dictionary containing the indeces of the lines of interest
        # text_lines is a list of the lines in the OCR output

        item_code_raw = text_lines[categories['Itemcode']]  # stores only the item code (from OCR)

        min_lev_item = 1000  # initialize to a value that cannot realistically be exceeded
        itemcode_cand_indeces = []  # for storing indeces!

        # use Levenshtein distance to find the item code(s) in the ASN that match the raw item code best
        for index in range(0, len(asn_data)):
            lev_dist = Levenshtein.distance(item_code_raw, asn_data[index][1])  # standard Levenshtein distance
            if lev_dist < min_lev_item:
                itemcode_cand_indeces = []  # reset the list to empty, and then collect the candidates
                itemcode_cand_indeces.append(index)  # store indeces for now
                min_lev_item = lev_dist
            elif lev_dist == min_lev_item:
                itemcode_cand_indeces.append(index)
            else:
                pass  # do nothing

        print "Smallest Lev. distance for item code : " + str(min_lev_item)

        # To confirm, could use the text description, and check if it give the same result as Lev dist on item codes
        min_lev_text = 1000
        text_cand_indeces = []

        # Again, apply Levenshtein distance
        text_raw = text_lines[categories['Description']]  # may or may not exist, depending on the outputinterpreter...

        for index in range(0, len(asn_data)):
            lev_dist = Levenshtein.distance(text_raw, asn_data[index][0])
            if lev_dist < min_lev_text:
                text_cand_indeces = []
                text_cand_indeces.append(index)
                min_lev_text = lev_dist
            elif lev_dist == min_lev_text:
                text_cand_indeces.append(index)

        print "Smallest Lev. distance for text: " + str(min_lev_text)
        itemcode_cand_values = []
        # check if the indeces are consistent between the approaches:
        if itemcode_cand_indeces == text_cand_indeces:
            for i in itemcode_cand_indeces:
                itemcode_cand_values.append(asn_data[i][1])
        # get the itemcode_candidates based on indeces

        return itemcode_cand_values, itemcode_cand_indeces

    def itemcodeFinderNoText(self, text_lines, categories):
        """Does not allow the text description to be used to identify the item code!
        Returns empty lists if no line identfied as the IC.
        The indeces returned refer to lines in the ASN."""
        asn = self.asn

        itemcode_cand_indeces = []
        candidate_values = []
        if type(categories['Itemcode']) is not bool:
            itemcode_raw = text_lines[categories['Itemcode']]
            min_lev_item = 1000

            for index in range(0, len(asn)):
                dist_ic = Levenshtein.distance(itemcode_raw, asn[index][1])

                if dist_ic < min_lev_item:
                    min_lev_item = dist_ic
                    itemcode_cand_indeces = []
                    itemcode_cand_indeces.append(index)
                    candidate_values = []
                    candidate_values.append(asn[index][1])
                elif dist_ic == min_lev_item:
                    itemcode_cand_indeces.append(index)
                    candidate_values.append(asn[index][1])

        return candidate_values, itemcode_cand_indeces

    def itemcodeFinder(self, text_lines, categories):
        """Returns the best guess(es) for the value of the item code, based on the ASN and the categorized lines dictionary
        Need to consider 4 cases: (0,0), (0,1) (1,0) and (1,1) where 0 is no, 1 is yes to the question if we have
        (itemcode_index, text_description_index)"""
        asn = self.asn

        itemcode_cand_indeces = []  # will later use this to find all the potential item codes
        text_description_cand_indeces = []

        if (type(categories['Itemcode']) is not bool) and (type(categories['Text description']) is not bool):
            itemcode_raw = text_lines[categories['Itemcode']]
            text_description_raw = text_lines[categories['Text description']]
            print "Found both IC and TD, proceeding accordingly!"
            min_lev_item = 1000
            min_lev_text = 1000

            for index in range(0, len(asn)):  # look at itemcode
                dist_ic = Levenshtein.distance(itemcode_raw, asn[index][1])
                dist_td = Levenshtein.distance(text_description_raw, asn[index][0])

                if dist_ic < min_lev_item:
                    min_lev_item = dist_ic
                    itemcode_cand_indeces = []  # re-initialize
                    itemcode_cand_indeces.append(index)
                elif dist_ic == min_lev_item:
                    itemcode_cand_indeces.append(index)

                if dist_td < min_lev_text:
                    text_description_cand_indeces = []
                    min_lev_text = dist_td
                    text_description_cand_indeces.append(index)
                elif dist_td == min_lev_text:
                    text_description_cand_indeces.append(index)
            # do cross-checking between the two lists
            candidate_values = []
            for k in itemcode_cand_indeces:
                if k in text_description_cand_indeces:
                    candidate_values.append(asn[k][1])

            print candidate_values

            return candidate_values, itemcode_cand_indeces  # may move the return statement

        elif (type(categories['Itemcode']) is not bool) and (type(categories['Text description']) == bool):
            itemcode_raw = text_lines[categories['Itemcode']]
            min_lev_item = 1000

            for index in range(0, len(asn)):
                dist_ic = Levenshtein.distance(itemcode_raw, asn[index][1])

                if dist_ic < min_lev_item:
                    itemcode_cand_indeces = []
                    min_lev_item = dist_ic
                    itemcode_cand_indeces.append(index)
                elif dist_ic == min_lev_item:
                    itemcode_cand_indeces.append(index)

            candidate_values = []
            for j in range(0, len(itemcode_cand_indeces)):
                candidate_values.append(asn[itemcode_cand_indeces[j]][1])

            print candidate_values

            return candidate_values, itemcode_cand_indeces  # may move

        elif (type(categories['Itemcode']) == bool) and (type(categories['Text description']) is not bool):
            text_description_raw = text_lines[categories['Text description']]
            min_lev_text = 1000

            for index in range(0, len(asn)):
                dist_td = Levenshtein.distance(text_description_raw, asn[index][0])

                if dist_td < min_lev_text:
                    text_description_cand_indeces = []
                    min_lev_text = dist_td
                    text_description_cand_indeces.append(index)
                elif dist_td == min_lev_text:
                    text_description_cand_indeces.append(index)

            candidate_values = []
            for j in range(0, len(text_description_cand_indeces)):
                candidate_values.append(asn[itemcode_cand_indeces[j]][1])
            print candidate_values

            return candidate_values, itemcode_cand_indeces

        else:
            print "Did not find neither itemcode nor text description. Consider changing image processing method "
            return False

    def solidassortFinderPerfect(self, text_lines, categories, dist):  # this is where everything gets messed up
        asn = self.asn
        text_lines = text_lines
        categories = categories
        dist = dist
        dist = 1000
        # only works if the method for calculating distances is the same, and if the index for sa is correct.
        solidassort = text_lines[categories['Solidcode']].replace(" ", "")
        solidassort = solidassort.replace("\n", "")
        solidassort_indeces = []

        # iteration 1: find the minimal cost
        # pass 2: find the value corresponding to the minimum cost

        for k in range(1, len(asn)):
            if WagnerFischer(solidassort, asn[k][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost < dist:
                solidassort_indeces = []
                solidassort_indeces.append(k)
                dist = WagnerFischer(solidassort, asn[k][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost
            elif WagnerFischer(solidassort, asn[k][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost == dist:
                solidassort_indeces.append(k)

        for k in solidassort_indeces:
            print asn[k][2]

        return solidassort_indeces

    def solidassortFinder(self, text_lines, categories, itemcode_indeces):
        asn = self.asn

        solidassort = text_lines[categories['Solidcode']]

        min_dist = 1000
        min_indeces = []

        if type(categories['Solidcode']) is not bool:
            for k in itemcode_indeces:  # assume that itemcode is correctly identified. Will introduce some bias, but will be similar to how the problem is actually solved
                if WagnerFischer(solidassort, asn[k][2], deletion=DELETIONNOCOST).cost < min_dist:
                    min_dist = WagnerFischer(solidassort, asn[k][2], deletion=DELETIONNOCOST).cost
                    min_indeces = []
                    min_indeces.append(k)
                elif WagnerFischer(solidassort, asn[k][2], deletion=DELETIONNOCOST).cost == min_dist:
                    min_indeces.append(k)

        return min_indeces

    def solidAssortReprocessed(self, text_line, itemcode_indeces):
        asn = self.asn
        line = text_line.replace(" ", "")
        min_dist = 10000
        min_indeces = []
        itemcode_indeces = itemcode_indeces

        for k in itemcode_indeces:
            if WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST).cost < min_dist:
                min_dist = WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST).cost
                min_indeces = []
                min_indeces.append(k)
            elif WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST).cost == min_dist:
                min_indeces.append(k)

        return min_indeces

    def saReprocessed(self, text_lines, itemcode_indices):
        asn = self.asn
        min_dist = 10000
        min_indeces = []
        itemcode_indeces = itemcode_indices

        "To-Do: instead of checking only the entire line for matches with the ASN, also do substrings"
        # Deletion cost = 0: any implications on the matching?
        # Insertion cost default

        for text_line in text_lines:
            line = text_line.replace(" ", "")
            line = line.replace("\n", "")

            for k in itemcode_indeces:
                if WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost < min_dist:
                    min_dist = WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost
                    min_indeces = []
                    min_indeces.append(k)
                elif WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost == min_dist:
                    min_indeces.append(k)

        return min_indeces, min_dist

    def saGlobal(self, solidassort_line):  # does not have modified subs costs
        text = solidassort_line  # may be one, or more lines
        asn = self.asn

        min_dist = 10000
        min_indices = []

        for text_line in text:
            line = text_line.replace(" ", "")
            line = line.replace("\n", "")

            for k in xrange(1, len(asn)):
                if WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST).cost < min_dist:
                    min_dist = WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST).cost
                    min_indices = []
                    min_indices.append(k)
                elif WagnerFischer(line, asn[k][2], deletion=DELETIONNOCOST).cost == min_dist:
                    min_indices.append(k)

        return min_indices, min_dist

    def solidorassort(self, line):  # find out if the line is solid or assort code (or at least which is most likely. Assume img processing went well
        """Important notice: The assort code can also contain double 0, so need to also use length to check
        Assumes there will be 000 at the end of the solid code!"""
        line = line
        line = line.strip('\n')
        line = line.upper()
        line = line.replace('O', '0')
        line = line.replace(' ', '')
        # convert O's to 0's
        if '000' or '00' in line:
            if len(line) > 4:
                return 'solid'
            else:
                return 'assort'
        else:
            return 'assort'

    def sigdigsSolid(self, candidate_indeces):  # assumes the solid code has 8 digits
        asn_data = self.asn_data

        solidcodes = []
        updated_indeces = []
        for index in candidate_indeces:
            if len(asn_data[index][2]) > 4:
                solidcodes.append(asn_data[index][2])
                updated_indeces.append(index)
        # the assort codes would also be included in the above list... need to remove based on len()

        sigdig_list = []
        for x in range(0, 8):
            sigdig_list.append(False)  # initialize to zero (false)

        # iterate through each of the 8 positions to find out if they are significant or not
        # note that normally the last 3 positions should NEVER be significant (all zeros)
        for i in range(0, 8):  # go through 0 to 7 (8 positions)
            digits_seen = []
            for j in range(0, len(solidcodes)):
                if solidcodes[j][i] not in digits_seen:  # check for any problems caused by difference in datatype
                    digits_seen.append(solidcodes[j][i])
            if len(digits_seen) > 1:
                sigdig_list[i] = True

        return sigdig_list, solidcodes, updated_indeces

    def sigdigsAssort(self, candidate_indeces):
        """Candidate indeces: Based on itemcodeFinder, the indeces with item code(s) that are
         considered most likely to correspond to the box.
         Returns a list of the significant characters [bool, bool, bool, bool] where True is significant"""

        asn = self.asn

        candidate_indeces = candidate_indeces
        assortcodes = []
        updated_indeces = []
        for index in candidate_indeces:
            if len(asn[index][2]) == 4:  # is an assort code
                assortcodes.append(asn[index][2])
                updated_indeces.append(index)

        sigdit_asort_list = [False, False, False, False]

        for i in range(0, 4):
            digits_seen = []
            for j in range(0, len(assortcodes)):
                if assortcodes[j][i] not in digits_seen:
                    digits_seen.append(assortcodes[j][i])
            if len(digits_seen) > 1:
                sigdit_asort_list[i] = True

        return sigdit_asort_list, assortcodes, updated_indeces

    def boxFinder(self, text_lines, categories, img, boxes):  # objective is to uniquely determine which box the image is from, if not sure then return > 1 line. Is this outdated?
        """Not: It'd be possible to use the ASN to check whether it's applicable at all to have solid/assort
        But if the output is too much nonsense then it doesn't really matter...
        The keywords in categories are Itemcode, Solidcode, Assortcode, Pack size, Text description"""
        asn_data = self.asn_data
        itemcode_cand_values, itemcode_cand_indeces = self.itemcodeFinder(text_lines=text_lines, categories=categories)  # look into this
        # text_lines contains the OCR output. Categories obvious. Have self.asn_data still from the initialization
        # next step is to: find out if is solid or assort, then identify critical digits
        if len(asn_data) == 1:  # 1 should be adjusted based on how many nonsense lines are in the ASN
            pass  # special case, only applicable to when there's only a single line left in the ASN..
        solid_or_assort = self.solidorassort(text_lines[categories['Solidcode']])  # ret 'solid' or 'assort'

        if solid_or_assort in 'solid':  # for solid case
            sigdig_list, solidcodes, updated_indeces = self.sigdigsSolid(itemcode_cand_indeces)
            print sigdig_list
            extractor = charextractor(img, boxes)
            extract_res = extractor.extractSolid(categories['Solidcode'], sigdig_list)
            # Image.fromarray(extract_res).show()  # is it array or image format?
            Image.fromarray(extract_res).save("solidslice.png")
            print "Done - for now "
            # check which digits are significant

        return solidcodes, updated_indeces

    def checkerOutdated(self, text_lines, ind_dictionary):  # to-do: check for internal consistencies when assigning indeces to categories
        # testing sequence: Itemcode, Solid/assort code, Text description.
        # intialize the lists for result collection:
        asn = self.asn
        result_list = []
        for k in range(0, len(text_lines)):
            result_list.append([False, False, False])

        """sub_penalties = np.ones((128, 128), dtype=np.float64)  # for weighted Lev. (if ever applicable)
        ins_penalties = np.ones((128, 128), dtype=np.float64)*5
        del_penalties = np.ones((128, 128), dtype=np.float64)*3
        sub_hacks = np.zeros((128, 128), dtype=np.float64)"""

        # test the lines against ASN, not concerned about the values of the best matches yet, just want to confirm line types
        for index in range(0, len(text_lines)):  # do a loophole to find the substitution-based score..may lead to negative values???
            cur_min_ic = 1000
            cur_min_sa = 1000
            cur_min_td = 1000
            for element_no in range(0, len(asn)):
                dist_ic = WagnerFischer(text_lines[index], asn[element_no][1]).cost
                checking_sa = text_lines[index].replace("-", "")
                dist_sa = WagnerFischer(checking_sa, asn[element_no][2]).cost
                dist_td = WagnerFischer(text_lines[index], asn[element_no][0]).cost
                if dist_ic < cur_min_ic:
                    result_list[index][0] = dist_ic
                    cur_min_ic = dist_ic
                if dist_sa < cur_min_sa:
                    result_list[index][1] = dist_sa
                    cur_min_sa = dist_sa
                if dist_td < cur_min_td:
                    cur_min_td = dist_td
                    result_list[index][2] = dist_td

        print result_list

        # process result_list to make a dictionary the indeces of useful lines
        result_list_dict = {'Itemcode': False, 'Solidcode': False, 'Text description': False}

        min_dist_ic = 1000
        min_dist_sa = 1000
        min_dist_td = 1000

        index_ic = False
        index_sa = False
        index_td = False

        for x in range(0, len(result_list)):  # if tied, go by index heuristics

            if result_list[x][0] < min_dist_ic:
                min_dist_ic = result_list[x][0]
                index_ic = x
            if result_list[x][1] < min_dist_sa:
                min_dist_sa = result_list[x][1]
                index_sa = x
            if result_list[x][2] < min_dist_td:
                min_dist_td = result_list[x][2]
                index_td = x

        result_list_dict['Itemcode'] = index_ic
        result_list_dict['Solidcode'] = index_sa
        result_list_dict['Text description'] = index_td

        # compare the results to the conclusions reached by the outputInterpreter(). Compare result_list and ind_dictionary
        ind_dictionary = ind_dictionary
        final_ind_dict = {'Itemcode': False, 'Solidcode': False, 'Text description': False}
        if ind_dictionary['Itemcode'] == result_list_dict['Itemcode']:  # base case, can safely conclude
            print "Itemcode confirmed on line: " + str(ind_dictionary['Itemcode']) + " -- " + str(text_lines[ind_dictionary['Itemcode']])
            final_ind_dict['Itemcode'] = ind_dictionary['Itemcode']
        elif result_list_dict['Itemcode'] > 0 and ind_dictionary['Itemcode'] > 0:
            final_ind_dict['Itemcode'] = min(ind_dictionary['Itemcode'], result_list_dict['Itemcode'])  # if include specific default value, then would usually get picked assuming noise is more common than omission
        else:
            final_ind_dict['Itemcode'] = max(ind_dictionary['Itemcode'], result_list_dict['Itemcode'])

        if ind_dictionary['Solidcode'] == result_list_dict['Solidcode']:
            print "Solidcode confirmed on line: " + str(ind_dictionary['Solidcode']) + " -- " + str(text_lines[ind_dictionary['Solidcode']])
            final_ind_dict['Solidcode'] = ind_dictionary['Solidcode']
        elif result_list_dict['Solidcode'] > 0 and ind_dictionary['Solidcode'] > 0:
            final_ind_dict['Solidcode'] = min(ind_dictionary['Solidcode'], result_list_dict['Solidcode'])
        else:
            final_ind_dict['Solidcode'] = max(ind_dictionary['Solidcode'], result_list_dict['Solidcode'])

        if ind_dictionary['Text description'] == result_list_dict['Text description']:
            print "Text description confirmed on line: " + str(ind_dictionary['Text description']) + " -- " + str(text_lines[ind_dictionary['Text description']])
            final_ind_dict['Text description'] = ind_dictionary['Text description']
        elif result_list_dict['Text description'] > 0 and ind_dictionary['Text description'] > 0:
            final_ind_dict['Text description'] = min(ind_dictionary['Text description'], result_list_dict['Text description'])
        else:
            final_ind_dict['Text description'] = max(ind_dictionary['Text description'], result_list_dict['Text description'])
        print ind_dictionary
        print result_list_dict
        # To-Do: avoid inconsistencies and duplicates
        self.itemcodeFinder(text_lines, final_ind_dict)
        return final_ind_dict, text_lines

    def checkerNoText(self, text_lines, ind_dictionary):
        # testing sequence: Itemcode, Solid/assort code, Text description.
        # intialize the lists for result collection:
        asn = self.asn
        skip_segmentation = False
        result_list = []
        for k in range(0, len(text_lines)):
            result_list.append([False, False, False])

        dist_threshold = 0.25

        # test the lines against ASN, not concerned about the values of the best matches yet, just want to confirm line types
        for index in range(0, len(
                text_lines)):  # do a loophole to find the substitution-based score..may lead to negative values???
            cur_min_ic = 1000
            cur_min_sa = 1000
            cur_min_td = 1000
            for element_no in range(1, len(asn)):
                dist_ic = 1.0 * WagnerFischer(text_lines[index], asn[element_no][1], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost
                #dist_ic = 1.0 * WagnerFischer(text_lines[index], asn[element_no][1],
                                              #deletion=DELETIONSMATCHER).cost / max(len(asn[element_no][1]), 1)
                # dist_ic = 1.0*Levenshtein.distance(text_lines[index], asn[element_no][1])/max(len(asn[element_no][1]), 1)
                if len(asn[element_no][1]) == 0 or len(text_lines[index]) == 0:
                    dist_ic = 999

                checking_sa = text_lines[index]
                dist_sa = 1.0 * WagnerFischer(checking_sa, asn[element_no][2], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost
                #dist_sa = 1.0 * WagnerFischer(checking_sa, asn[element_no][2], deletion=DELETIONNOCOST).cost / max(  # originally DELETIIONSMATCHER
                    #len(asn[element_no][2]), 1)
                # dist_sa = 1.0*Levenshtein.distance(checking_sa, asn[element_no][2])/max(len(asn[element_no][2]), 1)
                if len(asn[element_no][2]) == 0 or len(text_lines[index]) == 0:
                    dist_sa = 999

                # would not use the dist_td for this method (checkerNoText)?
                dist_td = 1.0 * WagnerFischer(text_lines[index], asn[element_no][0], deletion=DELETIONNOCOST, substitution=SUBSTITUTION).cost
                #dist_td = 1.0 * WagnerFischer(text_lines[index], asn[element_no][0],
                                              #deletion=DELETIONSMATCHER).cost / max(len(asn[element_no][0]), 1)
                # dist_td = 1.0*Levenshtein.distance(text_lines[index], asn[element_no][0])/max(len(asn[element_no][0]), 1)
                if len(asn[element_no][0]) == 0 or len(text_lines[index]) == 0:
                    dist_td = 999

                if dist_ic < cur_min_ic:
                    result_list[index][0] = dist_ic
                    cur_min_ic = dist_ic
                if dist_sa < cur_min_sa:
                    result_list[index][1] = dist_sa
                    cur_min_sa = dist_sa
                if dist_td < cur_min_td:
                    cur_min_td = dist_td
                    result_list[index][2] = dist_td

        print result_list

        # process result_list to make a dictionary the indeces of useful lines
        result_list_dict = {'Itemcode': False, 'Solidcode': False, 'Text description': False}

        min_dist_ic = 1000
        min_dist_sa = 1000
        min_dist_td = 1000

        index_ic = False
        index_sa = False
        index_td = False
        # find the best scorers (based on ASN) for each category
        for x in range(0, len(result_list)):  # if tied, go by index heuristics

            if result_list[x][0] < min_dist_ic:
                min_dist_ic = result_list[x][0]
                index_ic = x
            if result_list[x][1] < min_dist_sa:
                min_dist_sa = result_list[x][1]
                index_sa = x
            if result_list[x][2] < min_dist_td:
                min_dist_td = result_list[x][2]
                index_td = x

        if index_ic == index_td:
            index_td = False

        if index_ic == index_sa:
            index_sa = False

        result_list_dict['Itemcode'] = index_ic
        result_list_dict['Solidcode'] = index_sa
        result_list_dict['Text description'] = index_td

        # compare the results to the conclusions reached by the outputInterpreter(). Compare result_list and ind_dictionary
        ind_dictionary = ind_dictionary
        final_ind_dict = {'Itemcode': False, 'Solidcode': False, 'Text description': False}

        if type(ind_dictionary['Itemcode']) is bool and type(result_list_dict['Itemcode']) is bool:
            final_ind_dict['Itemcode'] = False
        elif type(ind_dictionary['Itemcode']) is bool and type(result_list_dict['Itemcode']) is not bool:
            final_ind_dict['Itemcode'] = result_list_dict['Itemcode']
        elif type(ind_dictionary['Itemcode']) is not bool and type(result_list_dict['Itemcode']) is bool:
            final_ind_dict['Itemcode'] = ind_dictionary['Itemcode']
        else:
            final_ind_dict['Itemcode'] = min(ind_dictionary['Itemcode'], result_list_dict['Itemcode'])

        if type(ind_dictionary['Solidcode']) is bool and type(result_list_dict['Solidcode']) is bool:
            final_ind_dict['Solidcode'] = False
        elif type(ind_dictionary['Solidcode']) is bool and type(result_list_dict['Solidcode']) is not bool:
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']
        elif type(ind_dictionary['Solidcode']) is not bool and type(result_list_dict['Solidcode']) is bool:
            final_ind_dict['Solidcode'] = ind_dictionary['Solidcode']
        else:
            final_ind_dict['Solidcode'] = min(ind_dictionary['Solidcode'], result_list_dict['Solidcode'])

        if type(ind_dictionary['Text description']) is bool and type(result_list_dict['Text description']) is bool:
            final_ind_dict['Text description'] = False
        elif type(ind_dictionary['Text description']) is bool and type(
                result_list_dict['Text description']) is not bool:
            final_ind_dict['Text description'] = result_list_dict['Text description']
        elif type(ind_dictionary['Text description']) is not bool and type(
                result_list_dict['Text description']) is bool:
            final_ind_dict['Text description'] = ind_dictionary['Text description']
        else:
            final_ind_dict['Text description'] = min(ind_dictionary['Text description'],
                                                     result_list_dict['Text description'])

        print ind_dictionary
        print result_list_dict
        print final_ind_dict

        if min_dist_ic < dist_threshold:
            final_ind_dict['Itemcode'] = result_list_dict['Itemcode']

        if min_dist_sa < dist_threshold:
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']

        if min_dist_td < dist_threshold:
            final_ind_dict['Text description'] = result_list_dict['Text description']

        if type(ind_dictionary['Text description']) is bool:
            final_ind_dict['Text description'] = False

        if min_dist_sa == 0.0: # may want to be more flexible here, or change the way min_dist_sa is computed
            print "Perfect on S/A!"
            skip_segmentation = True
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']
            itemcode_values, itemcode_indeces = self.itemcodeFinderNoText(text_lines,
                                                                          final_ind_dict)  # forward this later!
            solidassort_indeces = self.solidassortFinderPerfect(text_lines, final_ind_dict, min_dist_sa)
        elif min_dist_sa < 0.25:
            print "Satisfactory on S/A!"
            skip_segmentation = False
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']
            itemcode_values, itemcode_indeces = self.itemcodeFinderNoText(text_lines,
                                                                          final_ind_dict)  # forward this later!
            solidassort_indeces = self.solidassortFinderPerfect(text_lines, final_ind_dict, min_dist_sa)
        else:
            itemcode_values, itemcode_indeces = self.itemcodeFinderNoText(text_lines,
                                                                          final_ind_dict)  # forward this later!
            solidassort_indeces = self.solidassortFinder(text_lines, final_ind_dict, itemcode_indeces)
        print final_ind_dict

        unique_ic = True
        itemcode_check_uniques = itemcode_values
        # check if the item code values are unique.
        while len(itemcode_check_uniques) > 1:
            element = itemcode_check_uniques.pop()
            if element not in itemcode_check_uniques:
                print "Not unique item code! "
                unique_ic = False
        # print "Remember to forward this output to the next step ! "

        return final_ind_dict, text_lines, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces

    def checker(self, text_lines, ind_dictionary):
        # testing sequence: Itemcode, Solid/assort code, Text description.
        # intialize the lists for result collection:
        asn = self.asn
        skip_segmentation = False
        result_list = []
        for k in range(0, len(text_lines)):
            result_list.append([False, False, False])

        """sub_penalties = np.ones((128, 128), dtype=np.float64)
        ins_penalties = np.ones((128, 128), dtype=np.float64)*5
        del_penalties = np.ones((128, 128), dtype=np.float64)*3
        sub_hacks = np.zeros((128, 128), dtype=np.float64)"""

        dist_threshold = 0.30

        # test the lines against ASN, not concerned about the values of the best matches yet, just want to confirm line types
        for index in range(0, len(text_lines)):  # do a loophole to find the substitution-based score..may lead to negative values???
            cur_min_ic = 1000
            cur_min_sa = 1000
            cur_min_td = 1000
            for element_no in range(1, len(asn)):

                dist_ic = 1.0*WagnerFischer(text_lines[index], asn[element_no][1], deletion=DELETIONSMATCHER).cost/max(len(asn[element_no][1]), 1)
                # dist_ic = 1.0*Levenshtein.distance(text_lines[index], asn[element_no][1])/max(len(asn[element_no][1]), 1)
                if len(asn[element_no][1]) == 0 or len(text_lines[index]) == 0:
                    dist_ic = 999

                checking_sa = text_lines[index]
                dist_sa = 1.0 * WagnerFischer(checking_sa, asn[element_no][2], deletion=DELETIONSMATCHER).cost/max(len(asn[element_no][2]), 1)
                # dist_sa = 1.0*Levenshtein.distance(checking_sa, asn[element_no][2])/max(len(asn[element_no][2]), 1)
                if len(asn[element_no][2]) == 0 or len(text_lines[index]) == 0:
                    dist_sa = 999

                dist_td = 1.0*WagnerFischer(text_lines[index], asn[element_no][0], deletion=DELETIONSMATCHER).cost/max(len(asn[element_no][0]), 1)
                # dist_td = 1.0*Levenshtein.distance(text_lines[index], asn[element_no][0])/max(len(asn[element_no][0]), 1)
                if len(asn[element_no][0]) == 0 or len(text_lines[index]) == 0:
                    dist_td = 999

                if dist_ic < cur_min_ic:
                    result_list[index][0] = dist_ic
                    cur_min_ic = dist_ic
                if dist_sa < cur_min_sa:
                    result_list[index][1] = dist_sa
                    cur_min_sa = dist_sa
                if dist_td < cur_min_td:
                    cur_min_td = dist_td
                    result_list[index][2] = dist_td

        print result_list

        # process result_list to make a dictionary the indeces of useful lines
        result_list_dict = {'Itemcode': False, 'Solidcode': False, 'Text description': False}

        min_dist_ic = 1000
        min_dist_sa = 1000
        min_dist_td = 1000

        index_ic = False
        index_sa = False
        index_td = False
        # find the best scorers (based on ASN) for each category
        for x in range(0, len(result_list)):  # if tied, go by index heuristics

            if result_list[x][0] < min_dist_ic:
                min_dist_ic = result_list[x][0]
                index_ic = x
            if result_list[x][1] < min_dist_sa:
                min_dist_sa = result_list[x][1]
                index_sa = x
            if result_list[x][2] < min_dist_td:
                min_dist_td = result_list[x][2]
                index_td = x

        if index_ic == index_td:
            index_td = False

        if index_ic == index_sa:
            index_sa = False

        result_list_dict['Itemcode'] = index_ic
        result_list_dict['Solidcode'] = index_sa
        result_list_dict['Text description'] = index_td

        # compare the results to the conclusions reached by the outputInterpreter(). Compare result_list and ind_dictionary
        ind_dictionary = ind_dictionary
        final_ind_dict = {'Itemcode': False, 'Solidcode': False, 'Text description': False}

        if type(ind_dictionary['Itemcode']) is bool and type(result_list_dict['Itemcode']) is bool:
            final_ind_dict['Itemcode'] = False
        elif type(ind_dictionary['Itemcode']) is bool and type(result_list_dict['Itemcode']) is not bool:
            final_ind_dict['Itemcode'] = result_list_dict['Itemcode']
        elif type(ind_dictionary['Itemcode']) is not bool and type(result_list_dict['Itemcode']) is bool:
            final_ind_dict['Itemcode'] = ind_dictionary['Itemcode']
        else:
            final_ind_dict['Itemcode'] = min(ind_dictionary['Itemcode'], result_list_dict['Itemcode'])

        if type(ind_dictionary['Solidcode']) is bool and type(result_list_dict['Solidcode']) is bool:
            final_ind_dict['Solidcode'] = False
        elif type(ind_dictionary['Solidcode']) is bool and type(result_list_dict['Solidcode']) is not bool:
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']
        elif type(ind_dictionary['Solidcode']) is not bool and type(result_list_dict['Solidcode']) is bool:
            final_ind_dict['Solidcode'] = ind_dictionary['Solidcode']
        else:
            final_ind_dict['Solidcode'] = min(ind_dictionary['Solidcode'], result_list_dict['Solidcode'])

        if type(ind_dictionary['Text description']) is bool and type(result_list_dict['Text description']) is bool:
            final_ind_dict['Text description'] = False
        elif type(ind_dictionary['Text description']) is bool and type(result_list_dict['Text description']) is not bool:
            final_ind_dict['Text description'] = result_list_dict['Text description']
        elif type(ind_dictionary['Text description']) is not bool and type(result_list_dict['Text description']) is bool:
            final_ind_dict['Text description'] = ind_dictionary['Text description']
        else:
            final_ind_dict['Text description'] = min(ind_dictionary['Text description'], result_list_dict['Text description'])

        print ind_dictionary
        print result_list_dict
        print final_ind_dict

        if min_dist_ic < dist_threshold:
            final_ind_dict['Itemcode'] = result_list_dict['Itemcode']

        if min_dist_sa < dist_threshold:
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']

        if min_dist_td < dist_threshold:
            final_ind_dict['Text description'] = result_list_dict['Text description']

        if type(ind_dictionary['Text description']) is bool:
            final_ind_dict['Text description'] = False

        if min_dist_sa == 0.0:
            print "Perfect on S/A!"
            skip_segmentation = True
            final_ind_dict['Solidcode'] = result_list_dict['Solidcode']

        print final_ind_dict

        itemcode_values, itemcode_indeces = self.itemcodeFinder(text_lines, final_ind_dict)  # forward this later!

        if skip_segmentation:
            solidassort_indeces = self.solidassortFinderPerfect(text_lines, final_ind_dict, min_dist_sa)  # need to feed the minimum distance
        else:
            solidassort_indeces = self.solidassortFinder(text_lines, final_ind_dict, itemcode_indeces)

        unique_ic = True
        itemcode_check_uniques = itemcode_values
        # check if the item code values are unique.
        while len(itemcode_check_uniques) > 1:
            element = itemcode_check_uniques.pop()
            if element not in itemcode_check_uniques:
                print "Not unique item code! "
                unique_ic = False
        # print "Remember to forward this output to the next step ! "

        return final_ind_dict, text_lines, itemcode_indeces, unique_ic, skip_segmentation, solidassort_indeces


class charextractor():
    def __init__(self, img, boxes):
        self.img = img  # will be binary
        self.boxes = boxes

    def extractSolid(self, index_solid, sigdig_arr):  # after knowing which digits are significant figures. Need to pass on what we know about the line's pos and which figs are sigs
        """Extract only the bounding rectangle of the line corresponding to solid code, all other pixels are discarded
        Use either cv2.canny or cv2.findContours() then deal with each cont. individually.
        Ideally, want to skip the irrelevant conts. First: extract the entire solid code region. """
        index_solid = index_solid
        sigdig_arr = sigdig_arr
        img = self.img
        boxes = self.boxes

        edge_adjustment = 10  # pixels  # introduces issues of out of bound of arrays in rare cases

        coordinates = boxes[index_solid]
        x_top_left = max(coordinates[0] - edge_adjustment, 0)
        y_top_left = max(coordinates[1] - edge_adjustment, 0)
        x_bottom_right = min(coordinates[2] + edge_adjustment, img.shape[0])
        y_bottom_right = min(coordinates[3] + edge_adjustment, img.shape[1])
        # at this point, img is a 'JpegImageFile'
        # should extract more than just the bounding rectangle
        solidcode_img = img[y_top_left:y_bottom_right, x_top_left: x_bottom_right]

        return solidcode_img

    def extractAssort(self):
        pass


class outputInterpreter():

    def itemCand(self, line):
        # perform conversions
        line = line.replace(" ", "")
        line = line.replace("b", "6")
        line = line.replace("B", "8")
        line = line.upper()
        line = line.replace("O", "0")
        line = line.replace("I", "1")
        line = line.replace("L", "1")
        # print line
        # determine if the line is a candidate for item code or not
        five_or_more_digits_consecutive = re.findall(r"\d{5,20}", line, re.UNICODE)
        expression_xx_xx = re.findall(r"\d{2}-\d{2}", line, re.UNICODE)
        expression_xxx_xxxx = re.findall(r"\d{3}-\d{4,10}", line, re.UNICODE)
        # print five_or_more_digits_consecutive
        # print expression_xx_xx
        # print expression_xxx_xxxx

        digits = 0
        for c in line:
            if c.isdigit():
                digits += 1

        if digits > 8:
            nine_or_more_digits = True
        else:
            nine_or_more_digits = False

        if len(five_or_more_digits_consecutive) > 0 or len(expression_xx_xx) > 0 or len(expression_xxx_xxxx) or nine_or_more_digits:
            return True
        else:
            return False

    def solidCand(self, line):
        # perform conversions
        line = line.replace(" ", "")
        line = line.replace("b", "6")
        line = line.replace("B", "8")
        line = line.upper()
        line = line.replace("O", "0")
        line = line.replace("I", "1")
        line = line.replace("L", "1")
        # print line
        solidcode = False
        # determine is it's a candidate for solid code or not

        triple_zeros = re.findall(r"000", line, re.UNICODE)

        double_zeros_and_chars = re.search(r"00", line, re.UNICODE)  # first check if have double 00
        # next, based on the positions for the double zeros, want to check for other chars
        if double_zeros_and_chars:
            d = double_zeros_and_chars.start() > 1
            if d:
                solidcode = True

        format_xxx_hyphen = re.findall(r"-.{3}-", line, re.UNICODE)

        if len(re.findall(r"\d{8}", line, re.UNICODE)) > 0 or len(format_xxx_hyphen) > 0 or len(triple_zeros) > 0:
            solidcode = True

        return solidcode

    def assortCand(self, line):
        # format the string
        line = line.replace(" ", "")
        line = line.upper()
        # print line
        # look for formats satisfying the setup of assort code
        c_d_d = re.findall(r"[A-Z]{1}\d{2,4}", line, re.UNICODE)

        if len(c_d_d) > 0 or (0 < len(line) < 5):
            return True
        else:
            return False

    def packCand(self, line):
        # conversion of text
        line = line.replace(" ", "")
        line = line.upper()

        d_d_c_c = re.findall(r"\d{2}[A-Z]{2}", line, re.UNICODE)

        if len(d_d_c_c) > 0 or ("PC" in line or "PIECE" in line or "SIZE" in line or "PS" in line):
            return True
        else:
            return False

    def cartonCand(self, line):
        line = line.replace(" ", "")
        line = line.upper()

        if "NO" in line or "N0" in line or "NUM" in line or "CAR" in line or "CTN" in line or "CMO" in line or "CM0" in line:
            return True
        else:
            return False

    def descriptionCand(self, line):
        line = line.upper()

        spaces_around = re.findall(r"\s", line, re.UNICODE)

        line = line.replace(" ", "")

        six_or_more_chars = re.findall(r"[A-Z]{6,90}", line, re.UNICODE)

        if len(spaces_around) > 4 or len(six_or_more_chars) > 0:
            return True
        else:
            return False

    def categorizeLinesOutdated(self, ocr_output):
        """Input: String output from Tesseract
        Output: A dictionary containing the indeces for the useful pieces of information
        Assumption: No lines are missed"""

        dict_indeces = {'Itemcode': False, 'Solidcode': False, 'Assortcode': False, 'Pack size': False, 'Text description': False}  # initialize dictionary
        min_index_dict = {'Itemcode': 0, 'Solidcode': 1, 'Assortcode': 1,
                          'Pack size': 2, 'Carton number': 3, 'Text description': 4}
        max_index_dict = {'Itemcode': 100, 'Solidcode': 100, 'Assortcode': 100,
                          'Pack size': 100, 'Carton number': 100, 'Text description': 100}
        # step 1: split the ocr output to a list of lines
        list_lines = re.split('\n', ocr_output)

        candidate_info = []
        for x in range(0, len(list_lines)):
            candidate_info.append([])  # will store the candidate information in text format, and then use the keyword "in" to decide how to treat the lines later

        # first pass: find the candidate categories for each line
        k = 0
        for line in list_lines:

            if "SOLID" in line.upper():
                print "Have a line indicating the position of the solid code and item code!"
                max_index_dict['Itemcode'] = min(k-1, max_index_dict['Itemcode'])
                min_index_dict['Solidcode'] = max(k+1, min_index_dict['Solidcode'])
                min_index_dict['Assortcode'] = max(k+1, min_index_dict['Assortcode'])
                min_index_dict['Pack size'] = max(k+2, min_index_dict['Pack size'])
                min_index_dict['Carton number'] = max(k+3, min_index_dict['Carton number'])
                min_index_dict['Text description'] = max(k+4, min_index_dict['Text description'])

            if self.itemCand(line):
                print "Item code candidate found"
                candidate_info[k].append("Itemcode")

            if self.solidCand(line):
                print "Solid code candidate found"
                candidate_info[k].append("Solidcode")

            if self.assortCand(line):
                print "Assort code candidate found"
                candidate_info[k].append("Assortcode")

            if self.packCand(line):
                print "Pack size candidate found"
                candidate_info[k].append("Pack size")

            if self.cartonCand(line):
                print "Carton number candidate found"
                candidate_info[k].append("Carton number")

            if self.descriptionCand(line):
                print "Text description candidate found"
                candidate_info[k].append("Text description")

            k += 1
        print candidate_info

        # Start the second pass: Uniquely determining the indeces for each category

        # preparation stage: pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if min_index_dict[element] > x:
                    candidate_info[x].pop(j)
                j += 1
        print "After removing impossible cases, the new list is: "
        print candidate_info

        # Scenario 1: easy case
        pack_occur = 0
        pack_index = []
        j = 0
        for line in candidate_info:
            if "Pack size" in line:
                pack_occur += 1
                pack_index.append(j)
            j += 1

        if pack_occur == 1:  # look for the index of carton no
            dict_indeces['Pack size'] = pack_index[0]
            max_index_dict['Itemcode'] = pack_index[0]-1
            max_index_dict['Solidcode'] = pack_index[0]-1
            max_index_dict['Assortcode'] = pack_index[0]-1
            min_index_dict['Text description'] = max(min_index_dict['Text description'], pack_index[0]+2)

        carton_index = []
        j = 0
        for line in candidate_info:
            if "Carton number" in line:
                carton_index.append(j)
            j += 1

        if len(carton_index) == 1:
            max_index_dict['Itemcode'] = min(carton_index[0]-1, max_index_dict['Itemcode'])
            max_index_dict['Solidcode'] = min(carton_index[0]-1, max_index_dict['Solidcode'])
            max_index_dict['Assortcode'] = min(carton_index[0]-1, max_index_dict['Assortcode'])
            min_index_dict['Text description'] = max(min_index_dict['Text description'], carton_index[0]+1)

        # again, pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if min_index_dict[element] > x:
                    candidate_info[x].pop(j)
                j += 1
        print "After removing impossible cases, the new list is: "
        print candidate_info

        # pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if max_index_dict[element] < x:
                    candidate_info[x].pop(j)
                j += 1
        print "After removing impossible cases, the new list is: "
        print candidate_info

        # find the index of the text description line if there's a clear-cut case
        descr_count = 0
        descr_index = []
        j = 0
        for line in candidate_info:
            if "Text description" in line:
                descr_count += 1
                descr_index.append(j)
            j += 1

        if descr_count == 1:
            dict_indeces['Text description'] = descr_index[0]
            max_index_dict['Itemcode'] = min(max_index_dict['Itemcode'], descr_index[0]-4)
            max_index_dict['Solidcode'] = min(max_index_dict['Solidcode'], descr_index[0]-3)
            max_index_dict['Assortcode'] = min(max_index_dict['Assortcode'], descr_index[0]-3)
            max_index_dict['Pack size'] = min(max_index_dict['Pack size'], descr_index[0]-2)
            max_index_dict['Carton number'] = min(max_index_dict['Carton number'], descr_index[0]-1)
            max_index_dict['Text description'] = descr_index[0]
        # could do another looping through at this pt to eliminate impossible cases

            # again, pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if min_index_dict[element] > x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

            # pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if max_index_dict[element] < x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

        solid_count = 0
        solid_index = []
        j = 0
        for line in candidate_info:
            if "Solidcode" in line:
                solid_count += 1
                solid_index.append(j)
            j += 1

        if solid_count == 1:
            dict_indeces['Solidcode'] = solid_index[0]
            max_index_dict['Itemcode'] = min(solid_index[0]-1, max_index_dict['Itemcode'])
            min_index_dict['Pack size'] = max(min_index_dict['Pack size'], solid_index[0]+1)
            min_index_dict['Carton number'] = max(min_index_dict['Carton number'], solid_index[0]+2)
            min_index_dict['Text description'] = max(min_index_dict['Text description'], solid_index[0]+3)

            # again, pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if min_index_dict[element] > x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

            # pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if max_index_dict[element] < x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

        item_count = 0
        item_index = []
        j = 0

        for line in candidate_info:
            if "Itemcode" in line:
                item_count += 1
                item_index.append(j)
            j += 1

        if item_count == 1:
            dict_indeces['Itemcode'] = item_index[0]
            min_index_dict['Solidcode'] = max(min_index_dict['Solidcode'], item_index[0]+1)
            min_index_dict['Assortcode'] = max(min_index_dict['Assortcode'], item_index[0]+1)
            min_index_dict['Pack size'] = max(min_index_dict['Pack size'], item_index[0]+2)
            min_index_dict['Carton number'] = max(min_index_dict['Carton number'], item_index[0]+3)
            min_index_dict['Text description'] = max(min_index_dict['Text description'], item_index[0]+4)

        # based on min_index_dict and max_index_dict, infer the indeces of the unassigned lines
        for key, value in dict_indeces.iteritems():
            if not value:
                if min_index_dict[key] == max_index_dict[key]:
                    dict_indeces[key] = min_index_dict[key]
        # small issue: get both item and solid code this way, need a way to determine which of the two we have.
        # Can later use similar methods to previous in order to distinguish between the two categories

        print dict_indeces
        print candidate_info
        return list_lines, dict_indeces

    def categorizeLines(self, ocr_output):
        """Input: String output from Tesseract
        Output: A dictionary containing the indeces for the useful pieces of information
        Assumption: No lines are missed"""

        dict_indeces = {'Itemcode': False, 'Solidcode': False, 'Pack size': False, 'Text description': False}  # initialize dictionary
        min_index_dict = {'Itemcode': 0, 'Solidcode': 1,
                          'Pack size': 2, 'Carton number': 3, 'Text description': 4}
        max_index_dict = {'Itemcode': 100, 'Solidcode': 100,
                          'Pack size': 100, 'Carton number': 100, 'Text description': 100}
        # step 1: split the ocr output to a list of lines
        list_lines = re.split('\n', ocr_output)

        candidate_info = []
        for x in range(0, len(list_lines)):
            candidate_info.append([])  # will store the candidate information in text format, and then use the keyword "in" to decide how to treat the lines later

        # first pass: find the candidate categories for each line
        k = 0
        for line in list_lines:

            if "SOLID" in line.upper():
                # print "Have a line indicating the position of the solid code and item code!"
                max_index_dict['Itemcode'] = min(k-1, max_index_dict['Itemcode'])
                min_index_dict['Solidcode'] = max(k+1, min_index_dict['Solidcode'])
                min_index_dict['Pack size'] = max(k+2, min_index_dict['Pack size'])
                min_index_dict['Carton number'] = max(k+3, min_index_dict['Carton number'])
                min_index_dict['Text description'] = max(k+4, min_index_dict['Text description'])

            if self.itemCand(line):
                # print "Item code candidate found"
                candidate_info[k].append("Itemcode")

            if self.assortCand(line) or self.solidCand(line):
                    # print "Solid / Assort code candidate found"
                    candidate_info[k].append("Solidcode")

            if self.packCand(line):
                # print "Pack size candidate found"
                candidate_info[k].append("Pack size")

            if self.cartonCand(line):
                # print "Carton number candidate found"
                candidate_info[k].append("Carton number")

            if self.descriptionCand(line):
                # print "Text description candidate found"
                candidate_info[k].append("Text description")

            k += 1
        print candidate_info

        # Start the second pass: Uniquely determining the indeces for each category

        # preparation stage: pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if min_index_dict[element] > x:
                    candidate_info[x].pop(j)
                j += 1
        # print "After removing impossible cases, the new list is: "
        print candidate_info

        # Scenario 1: easy case
        pack_occur = 0
        pack_index = []
        j = 0
        for line in candidate_info:
            if "Pack size" in line:
                pack_occur += 1
                pack_index.append(j)
            j += 1

        if pack_occur == 1:  # look for the index of carton no
            dict_indeces['Pack size'] = pack_index[0]
            max_index_dict['Itemcode'] = pack_index[0]-1
            max_index_dict['Solidcode'] = pack_index[0]-1
            min_index_dict['Text description'] = max(min_index_dict['Text description'], pack_index[0]+2)

        carton_index = []
        j = 0
        for line in candidate_info:
            if "Carton number" in line:
                carton_index.append(j)
            j += 1

        if len(carton_index) == 1:
            max_index_dict['Itemcode'] = min(carton_index[0]-1, max_index_dict['Itemcode'])
            max_index_dict['Solidcode'] = min(carton_index[0]-1, max_index_dict['Solidcode'])
            min_index_dict['Text description'] = max(min_index_dict['Text description'], carton_index[0]+1)

        # again, pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if min_index_dict[element] > x:
                    candidate_info[x].pop(j)
                j += 1
        # print "After removing impossible cases, the new list is: "
        print candidate_info

        # pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if max_index_dict[element] < x:
                    candidate_info[x].pop(j)
                j += 1
        # print "After removing impossible cases, the new list is: "
        print candidate_info

        # find the index of the text description line if there's a clear-cut case
        descr_count = 0
        descr_index = []
        j = 0
        for line in candidate_info:
            if "Text description" in line:
                descr_count += 1
                descr_index.append(j)
            j += 1

        if descr_count == 1:
            dict_indeces['Text description'] = descr_index[0]
            max_index_dict['Itemcode'] = min(max_index_dict['Itemcode'], descr_index[0]-4)
            max_index_dict['Solidcode'] = min(max_index_dict['Solidcode'], descr_index[0]-3)
            max_index_dict['Pack size'] = min(max_index_dict['Pack size'], descr_index[0]-2)
            max_index_dict['Carton number'] = min(max_index_dict['Carton number'], descr_index[0]-1)
            max_index_dict['Text description'] = descr_index[0]
        # could do another looping through at this pt to eliminate impossible cases

            # again, pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if min_index_dict[element] > x:
                        candidate_info[x].pop(j)
                    j += 1
            # print "After removing impossible cases, the new list is: "
            print candidate_info

            # pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if max_index_dict[element] < x:
                        candidate_info[x].pop(j)
                    j += 1
            # print "After removing impossible cases, the new list is: "
            print candidate_info

        solid_count = 0
        solid_index = []
        j = 0
        for line in candidate_info:
            if "Solidcode" in line:
                solid_count += 1
                solid_index.append(j)
            j += 1

        if solid_count == 1:
            dict_indeces['Solidcode'] = solid_index[0]
            max_index_dict['Itemcode'] = min(solid_index[0]-1, max_index_dict['Itemcode'])
            min_index_dict['Pack size'] = max(min_index_dict['Pack size'], solid_index[0]+1)
            min_index_dict['Carton number'] = max(min_index_dict['Carton number'], solid_index[0]+2)
            min_index_dict['Text description'] = max(min_index_dict['Text description'], solid_index[0]+3)

            # again, pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if min_index_dict[element] > x:
                        candidate_info[x].pop(j)
                    j += 1
            # print "After removing impossible cases, the new list is: "
            print candidate_info

            # pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if max_index_dict[element] < x:
                        candidate_info[x].pop(j)
                    j += 1
            # print "After removing impossible cases, the new list is: "
            print candidate_info

        item_count = 0
        item_index = []
        j = 0

        for line in candidate_info:
            if "Itemcode" in line:
                item_count += 1
                item_index.append(j)
            j += 1

        if item_count == 1:
            dict_indeces['Itemcode'] = item_index[0]
            min_index_dict['Solidcode'] = max(min_index_dict['Solidcode'], item_index[0]+1)
            min_index_dict['Pack size'] = max(min_index_dict['Pack size'], item_index[0]+2)
            min_index_dict['Carton number'] = max(min_index_dict['Carton number'], item_index[0]+3)
            min_index_dict['Text description'] = max(min_index_dict['Text description'], item_index[0]+4)

        # based on min_index_dict and max_index_dict, infer the indeces of the unassigned lines
        for key, value in dict_indeces.iteritems():
            if not value:
                if min_index_dict[key] == max_index_dict[key]:
                    dict_indeces[key] = min_index_dict[key]
        # small issue: get both item and solid code this way, need a way to determine which of the two we have.
        # Can later use similar methods to previous in order to distinguish between the two categories

        print dict_indeces
        print candidate_info
        return list_lines, dict_indeces


if __name__ == "__main__":
    app = simpleapp_tk(None)  # No parent because first element
    app.title('HKUST x ABC Company')
    app.mainloop()  # Run infinite loop, waiting for events.
