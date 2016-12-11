""""coding: UTF-8"""
import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI

test_file = "Digits.png" # 500 of each digit, in order. Each image is 20x20. Get from google drive

img = cv2.imread(test_file)
gray = img

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # needed or not?

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells)

test = x[:, :100].reshape(-1, 400).astype(np.float32)

k = np.arange(10)
test_labels = np.repeat(k, 500)[:, np.newaxis]

index = 0
correct = 0

accs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

predictions = []
for lk in range(0, 10):
    predictions.append("")

with PyTessBaseAPI(psm = 10) as api:
    api.SetVariable("tessedit_char_whitelist", "0987654321")
    for l in x:
        for t in l:
            t = Image.fromarray(t)
            api.SetImage(t)
            text = api.GetUTF8Text().encode("utf-8")
            text = text.strip("\n")
            real = str(test_labels[index])
            a = int(test_labels[index])
            if text in real:
                correct += 1
                accs[int(test_labels[index])] += 1
            else:
                print text
                predictions[a] += text

            index += 1
            # print text
print correct
print index
accuracy = 100.0 * correct / 5000
print "Accuracy in %: " + str(accuracy)


print accs

for p in range(0, len(predictions)):
    print "The predictions for " + str(p) + " were: "
    a = predictions[p]
    print ''.join(sorted(a))
