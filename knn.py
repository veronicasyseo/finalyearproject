# based on: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
import numpy as np
import cv2
import pickle

# Prepare training data
train_file = "path to pickle"

train_data = pickle.load(open(train_file))

train_arr = np.array(train_data)
train_imgs, train_labels = np.split(train_arr, 2, axis = 1)

train_imgs = train_imgs.tolist()
train_imgs = np.asarray(train_imgs)
train_imgs = train_imgs.reshape(-1, 784).astype(np.float32) # flatten, 28x28=784


train_labels = train_labels.tolist()
train_labels = np.asarray(train_labels)
train_labels = train_labels.astype(np.float32)

# Carry out training
knn = cv2.ml.KNearest_create() # check later
knn.train(train_imgs, cv2.ml.ROW_SAMPLE, train_labels) # need to get correct input...

# Prepare test data
test_file = "path to another pickle"

test_data = pickle.load(open(test_file))

test_arr = np.array(test_data)

test_imgs, test_labels = np.split(test_arr, 2, axis = 1)
test_imgs = np.asarray(test_imgs)

test_imgs = test_imgs.tolist()
test_imgs = np.asarray(test_imgs)
test_imgs = test_imgs.reshape(-1, 784).astype(np.float32)

test_labels = test_labels.tolist()
test_labels = np.asarray(test_labels)
test_labels = test_labels.astype(np.float32)

# Test accuracy
ret, result, neighbours, dist = knn.findNearest(test_imgs, k = 3) # k may need to be changed

matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print "Overall accuracy: " + str(accuracy)

# Detailed accuracy information:
accuracies = [] # initialize
for number in range(0, 10):
    accuracies.append([0, 0])

for value in range(0, len(result)): # check
    cur_int = int(test_labels[value])
    accuracies[cur_int][0] += 1
    if result[value] == test_labels[value]:
        accuracies[cur_int][1] += 1

print accuracies

# Save the data
train_imgs = train_imgs.astype(np.uint8) # convert to take up less memory
train_labels = train_labels.astype(np.uint8) # convert back to float32 upon loading
np.savez('knn_data.npz', train = train_imgs, train_labels = train_labels)

# To load (for your reference):
"""with np.load('knn_data.npz') as data:
    print data.files # list the files stored
    train = data['train'].astype(np.float32)
    train_labels = data['train_labels'].astype(np.float32)"""
