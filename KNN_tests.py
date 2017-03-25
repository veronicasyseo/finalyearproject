"""THe goal is to explore
the relationship
betweeen accuracy, value of K and the number of training data"""

from tkFileDialog import askdirectory
import cv2
import numpy as np
import pickle
import os

# get new data
# directory = askdirectory()
directory = "/Users/sigurdandersberg/PycharmProjects/proj1/KNN_work/Pickles"

for filename in os.listdir(directory):
    if filename.endswith(".pickle"):  # check if is pickle or not
        train_data = pickle.load(open(os.path.join(directory, filename)))

        train_arr = np.array(train_data)
        train_imgs, train_labels = np.split(train_arr, 2, axis = 1)

        train_imgs = train_imgs.tolist()
        train_imgs = np.asarray(train_imgs)
        train_imgs = train_imgs.reshape(-1, 784).astype(np.float32) # flatten, 28x28=784

        train_labels = train_labels.tolist()
        train_labels = np.asarray(train_labels)
        train_labels = train_labels.astype(np.float32)

        knn_model_old = "/Users/sigurdandersberg/PycharmProjects/proj1/KNN_work/knn_data.npz"  # insert path here
        # obtain the old data
        with np.load(knn_model_old) as data:
            print data.files  # list the files stored
            train_old = data['train'].astype(np.float32)  # need to convert to correct data type as required by knn
            train_labels_old = data['train_labels'].astype(np.float32)
        print len(train_old)
        print len(train_labels_old)
        print len(train_imgs)
        print len(train_labels)
        # combine new and old data
        train_imgs = np.concatenate((train_old, train_imgs), 0)
        train_labels = np.append(train_labels_old, train_labels)
        # Carry out training
        knn = cv2.ml.KNearest_create()  # check later

        knn.train(train_imgs, cv2.ml.ROW_SAMPLE, train_labels)  # need to get correct input...

        # test the accuracy
        test_file = "/Users/sigurdandersberg/PycharmProjects/proj1/KNN_work/SD19_28x28_60000.pickle"

        test_data = pickle.load(open(test_file))

        test_arr = np.array(test_data)

        test_imgs, test_labels = np.split(test_arr, 2, axis=1)
        test_imgs = np.asarray(test_imgs)

        test_imgs = test_imgs.tolist()
        test_imgs = np.asarray(test_imgs)
        test_imgs = test_imgs.reshape(-1, 784).astype(np.float32)

        test_labels = test_labels.tolist()
        test_labels = np.asarray(test_labels)
        test_labels = test_labels.astype(np.float32)

        k_values = [1, 3, 5, 7, 9]

        for k_val in k_values:
            # Test accuracy
            ret, result, neighbours, dist = knn.findNearest(test_imgs, k=k_val)  # k may need to be changed

            matches = result == test_labels
            correct = np.count_nonzero(matches)
            accuracy = correct * 100.0 / result.size
            print "With this value of k: " + str(k_val)
            print "And with this number of training data: " + str(len(train_labels))
            print "Overall accuracy: " + str(accuracy)

            # Detailed accuracy information:
            accuracies = []  # initialize
            for number in range(0, 10):
                accuracies.append([0, 0])

            for value in range(0, len(result)):  # check
                cur_int = int(test_labels[value])
                accuracies[cur_int][0] += 1
                if result[value] == test_labels[value]:
                    accuracies[cur_int][1] += 1

            print accuracies

        # Save the data
        train_imgs = train_imgs.astype(np.uint8)  # convert to take up less memory
        train_labels = train_labels.astype(np.uint8)  # convert back to float32 upon loading
        np.savez('knn_data.npz', train=train_imgs, train_labels=train_labels)
