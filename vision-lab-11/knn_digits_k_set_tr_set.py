import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt 
img = cv.imread('digits.png')
if img is None:    #check if the image is valid and loaded
    sys.exit("Could not read the image")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
 
x = np.array(cells)
digits = x.reshape(10, 500, 20, 20)

k_values = list(range(1, 10, 1))
train_percents = list(range(10, 100, 10))
results = np.zeros((len(train_percents), len(k_values)))
for i , percent in enumerate(train_percents):
    n_train = int((percent / 100) * 500)  
    n_test = 500 - n_train

    train = digits[:, :n_train].reshape(-1, 400).astype(np.float32)
    test = digits[:, n_train:].reshape(-1, 400).astype(np.float32)

    train_labels = np.repeat(np.arange(10), n_train)[:, np.newaxis]
    test_labels = np.repeat(np.arange(10), n_test)[:, np.newaxis]
    for j, z in enumerate(k_values):
        knn = cv.ml.KNearest_create()
        knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

        ret, result, neighbours, dist = knn.findNearest(test, k=z)
        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        results[i, j] = accuracy
        print(f"K = {i}, Train: {percent}%, Test: {100 - percent}%  Accuracy: {accuracy:.2f}%")

for i, percent in enumerate(train_percents):
    plt.plot(k_values, results[i], marker='o', label=f"Train {percent}%")

plt.title("kNN Accuracy vs. k at Different Test/Train Splits")
plt.xlabel("k value")
plt.ylabel("Accuracy (%)")
plt.legend(title="Train split")
plt.grid(True)
plt.tight_layout()
plt.show()
