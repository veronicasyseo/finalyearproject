import csv
from PIL import Image
import numpy as np
from tesserocr import PyTessBaseAPI

inp = "letter.data" # downloaded from: http://ai.stanford.edu/~btaskar/ocr/

reading = []

count = 0
correct = 0

acc_arr = []

for x in range(0, 26):
    acc_arr.append([0, 0])


def inds(x):
    return {
        'a' : 0,
        'b' : 1,
        'c' : 2,
        'd' : 3,
        'e' : 4,
        'f' : 5,
        'g' : 6,
        'h' : 7,
        'i' : 8,
        'j' : 9,
        'k' : 10,
        'l' : 11,
        'm' : 12,
        'n' : 13,
        'o' : 14,
        'p' : 15,
        'q' : 16,
        'r' : 17,
        's' : 18,
        't' : 19,
        'u' : 20,
        'v' : 21,
        'w' : 22,
        'x' : 23,
        'y' : 24,
        'z' : 25,
    }[x]

with open(inp) as tsv:
    with PyTessBaseAPI(psm=10) as api:
        api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyz")  # only allow these characters
        for line in csv.reader(tsv, delimiter="\t"):
            print count
            count += 1
            pixels = line[6:133]
            pixels = np.resize(pixels, (16, 8))
            pixels = pixels.astype(dtype='uint8')
            img = Image.fromarray(255 * pixels)
            api.SetImage(img)
            text = api.GetUTF8Text().encode("utf-8")
            text = text.strip("\n")
            real = str(line[1])
            number = inds(real)
            acc_arr[number][0] += 1
            if text in real:
                correct += 1
                acc_arr[number][1] += 1
                print "Correct!"


accuracy = 100.0 * correct / count
print "Overall accuracy: " + str(accuracy)

print acc_arr # on the form [total, correct] for each letter, in order.



"""id: each letter is assigned a unique integer id [0]
letter: a-z [1]
next_id: id for next letter in the word, -1 if last letter [2]
word_id: each word is assigned a unique integer id (not used) [3]
position: position of letter in the word (not used) [4]
fold: 0-9 -- cross-validation fold [5]
p_i_j: 0/1 -- value of pixel in row i, column j [6] to [134] (133) 128 pixels, 16x8"""
