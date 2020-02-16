import pickle

def loadMNIST():
    """
    return: list: MNIST-Datensatz, geordnet
    """
    with open("MNIST_data_CSV//pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]
    image_size = 28 # width and length
    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size

    return [train_imgs,
    test_imgs,
    train_labels,
    test_labels,
    train_labels_one_hot,
    test_labels_one_hot]


