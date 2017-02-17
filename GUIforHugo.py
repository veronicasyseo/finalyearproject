#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
from tkFileDialog import askdirectory
import os
import numpy as np
from PIL import ImageTk, Image
from tesserocr import PyTessBaseAPI, RIL, PyResultIterator
from skimage.filters import threshold_adaptive
import cv2
import time

class simpleapp_tk(Tkinter.Tk):
    def __init__(self, parent):  # constructor
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent  # keep track of our parent
        self.next_wanted = False
        self.initialize()

    def DisplayImages(self, image_processed, filename):
        img2 = Image.open(filename)  # what if user cancelled?
        img2 = img2.resize((self.width, self.height), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img2)

        self.label1.config(image=img2)
        self.label1.image = img2

        image_processed = image_processed.resize((self.width, self.height), Image.ANTIALIAS)
        image_processed = ImageTk.PhotoImage(image_processed)
        self.label2.config(image=image_processed)
        self.label2.image = image_processed

    def DisplayOCRText(self, text):
        if len(text) > 400:
            text = "Use other image processing methods!"
        self.label3.config(text=text)

    def initialize(self):
        self.width = 500
        self.height = 500
        self.grid
        self.directory = askdirectory()
        self.filename = os.listdir(self.directory)
        self.index = 0

        button1 = Tkinter.Button(self, text=u"Next !", command=self.OnButtonClickNext)
        button1.grid(column=1, row=1)

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

        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)
        self.update()
        self.geometry(self.geometry())

    def OnButtonClickNext(self):

        if self.filename[self.index].endswith(".JPG"):

            img_read = cv2.imread(os.path.join(self.directory, self.filename[self.index]))
            area_lower_bound = 200  # originally 300
            print self.filename[self.index]  # Doesn't know the full path
            grayscale = cv2.cvtColor(img_read,
                                     cv2.COLOR_BGR2GRAY)  # potential improvement: using multiple color channels and combining results

            block_size = 201
            offset = 24
            binar_adaptive = threshold_adaptive(grayscale, block_size=block_size, offset=offset)

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

            img_for_tess = Image.fromarray(combined.astype('uint8') * 255)

            with PyTessBaseAPI(psm=6) as api:
                api.SetVariable("tessedit_char_whitelist",
                                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
                api.SetImage(img_for_tess)
                self.DisplayImages(api.GetThresholdedImage(), os.path.join(self.directory, self.filename[self.index]))
                self.DisplayOCRText(api.GetUTF8Text())
        self.index += 1

    def cyclethrough(self, index, filename, directory):

        directory = askdirectory()  # prompt user to select a folder

        for filename in os.listdir(directory):
            self.Stopper()
            if filename.endswith(".JPG"):

                img_read = cv2.imread(os.path.join(directory,filename))
                area_lower_bound = 200  # originally 300
                print filename  # Doesn't know the full path
                grayscale = cv2.cvtColor(img_read,
                                         cv2.COLOR_BGR2GRAY)  # potential improvement: using multiple color channels and combining results

                block_size = 201
                offset = 24
                binar_adaptive = threshold_adaptive(grayscale, block_size=block_size, offset=offset)

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

                img_for_tess = Image.fromarray(combined.astype('uint8') * 255)

                with PyTessBaseAPI(psm=6) as api:
                    api.SetVariable("tessedit_char_whitelist",
                                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
                    api.SetImage(img_for_tess)
                    self.DisplayImages(api.GetThresholdedImage(), os.path.join(directory, filename))
                    self.DisplayOCRText(api.GetUTF8Text())
                continue

            else:
                continue



if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('Hugo Accuracy Corporation')
    app.mainloop()  # Run infinite loop, waiting for events.
