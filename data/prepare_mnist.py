import numpy as np
import gzip
import pickle
# modified based on https://github.com/hsjeong5/MNIST-for-Numpy


filename = [["training_images", "train-images-idx3-ubyte.gz"], ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"], ["test_labels", "t10k-labels-idx1-ubyte.gz"]]

if __name__ == '__main__':
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")
