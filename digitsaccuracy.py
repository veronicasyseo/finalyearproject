""""coding: UTF-8"""
from PIL import Image
from tesserocr import PyTessBaseAPI
import pickle

# load data
test_file = "path to a pickle downloaded from google drive"
inputs = pickle.load(open(test_file)) # all the inputs, with labels.

# initialize variables 
index = 0
correct = 0
accs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
samples = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # in case of different number of samples for each digit
classifications = [] # prepare array for storing the incorrect predictions for each digit (string type)
for lk in range(0, 10):
    classifications.append("")

# Feed the images to tesseract, one by one. 
with PyTessBaseAPI(psm = 10) as api:
    api.SetVariable("tessedit_char_whitelist", "0987654321") # only allow these characters
    for l in inputs:
        t = Image.fromarray(l[0])
        api.SetImage(t)
        text = api.GetUTF8Text().encode("utf-8")
        text = text.strip("\n")
        real = str(l[1]) # string type for comparison
        a = int(l[1])
        samples[a] += 1 # count number of samples of each type
        if text in real:
            correct += 1
            accs[a] += 1
        else:
            classifications[a] += text

        index += 1

# printing results
print correct
print index
accuracy = 100.0 * correct / index
print "Accuracy in %: " + str(accuracy)
print "Individual number correct: "
print accs # the individual accuracies
print "Individual number of samples: "
print samples

# print incorrect classifications if wanted:
for p in range(0, len(classifications)):
    print "The incorrect classifications for " + str(p) + " were: "
    a = classifications[p]
    print ''.join(sorted(a))
