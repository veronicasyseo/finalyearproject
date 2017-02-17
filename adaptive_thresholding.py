from skimage.filters import threshold_adaptive
from tkFileDialog import askopenfilename
import cv2
from PIL import Image
from tesserocr import PyTessBaseAPI
import numpy as np

img = askopenfilename()

area_lower_bound = 200  # originally 300

img_read = cv2.imread(img)
grayscale = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)  # potential improvement: using multiple color channels and combining results 

block_size = 121
binar_adaptive = threshold_adaptive(grayscale, block_size, offset = 24)

# next, do noise removal
noisy = binar_adaptive.astype('uint8')*255

im2, contours, hierarchy = cv2.findContours(noisy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

large_contours = []

for cnt in contours:
    if cv2.contourArea(cnt) > area_lower_bound:
        large_contours.append(cnt)

black_bg = np.zeros((img_read.shape[0], img_read.shape[1]), dtype='uint8')
cv2.drawContours(black_bg, large_contours, -1, color=255, thickness=-1)
Image.fromarray(black_bg).show()  # black text on white background
combined = np.logical_and(255-black_bg, 255-noisy)  # why are some tiny pixels left here?

Image.fromarray(combined.astype('uint8')*255).show()

img_for_tess = Image.fromarray(combined.astype('uint8')*255)

with PyTessBaseAPI(psm=6) as api:
    api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321-.:/()")
    api.SetImage(img_for_tess)
    api.GetThresholdedImage().save('adaptive_thresh.png')
    print api.GetUTF8Text()

