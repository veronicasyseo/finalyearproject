""""coding: UTF-8"""

from PIL import Image
from tesserocr import PyTessBaseAPI
import pickle

# Load data:
input_file = "path to pickle" 
data = pickle.load(open(input_file))

# Initialize
count = 0
correct = 0
acc_arr = [] # array for counting individual samples and number of correct classifications
for x in range(0, 26): # for all the letters
    acc_arr.append([0, 0])

# For assigning index to each letter
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

# Testing accuracy
with PyTessBaseAPI(psm=10) as api:
    api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyz")
    for element in data:
        if str(element[1]).islower() and (str(element[1]) not in "0123456789"): # may be overkill but better safe than sorry
            index = inds(str(element[1]))
            acc_arr[index][0] += 1
            img = Image.fromarray(element[0])
            api.SetImage(img)
            text = api.GetUTF8Text().encode("utf-8")
            text = text.strip("\n")
            text = text.lower()
            if text in str(element[1]):
                correct += 1
                acc_arr[index][1] += 1

            count += 1

accuracy = 100.0 * correct / count
print "Overall accuracy: " + str(accuracy)

print "Individual results:"

print "Number of samples: "
for character in acc_arr:
    print character[0]

print "Correct: "
for character in acc_arr:
    print character[1]
