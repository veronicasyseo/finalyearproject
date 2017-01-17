#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
from PIL import ImageTk, Image
from tkFileDialog import askopenfilename
from tesserocr import PyTessBaseAPI, RIL, iterate_level
import numpy as np
import cv2
# import csv  # not used yet

class simpleapp_tk(Tkinter.Tk):
    def __init__(self, parent):  # constructor
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent  # keep track of our parent
        self.initialize()

    def initialize(self):  # will be used for creating all the widgets
        self.width = 400
        self.height = 400
        self.method = "Advanced"  # default
        self.grid()  # add things here later

        # button for loading input images
        button1 = Tkinter.Button(self, text=u"Load image !", command=self.OnButtonClickLoad)
        button1.grid(column=0, row=1)

        # button for loading Advance Shipment Notice
        button3 = Tkinter.Button(self, text=u"Load ASN (to be implemented)!", command=self.OnButtonClickASN)
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
        optionList = ["Advanced", "Basic", "ContourGaussianKernelOtsu", "GaussianKernelAndOtsu", "Otsu"]  # add more later
        self.dropVar = Tkinter.StringVar()
        self.dropVar.set("Advanced")  # default
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
        else:
            print "Method selection error!"

    def OnButtonClickASN(self):
        pass  # implement later

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
            api.SetImageFile(self.filename)
            text_output = api.GetUTF8Text()
            image_processed = api.GetThresholdedImage()
            return text_output, image_processed

    def Advanced(self):
        """Does contour analysis to find the likely text region(s) of an image,
         uses Otsu's binarization on the result
          and performs OCR with Tesseract"""
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
    pass  # implementation subject to format of Unloading Report

class outputInterpreter():
    pass  # implement later 

if __name__ == "__main__":
    app = simpleapp_tk(None)  # No parent because first element
    app.title('HKUST x ABC Company')
    app.mainloop()  # Run infinite loop, waiting for events.
