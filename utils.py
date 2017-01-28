import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


def preprocess(imgs):
    """
    input: takes in numpy array representing images
    output: cleaned and processed data
    """

    img_mean = np.mean(imgs, axis=0)
    img_std = np.std(imgs, axis=0)

    normalized = imgs - img_mean / img_std

    return normalized


def load_images(path, label):
    """
    input: file path containing images, and the labes it should be assigned (1 or 0)
    output: the images converted into a numpy matrix
    """

    dims=64

    files = [path + '/' + f for f in os.listdir(path) if ".jpg" in f]
    imgs = [imresize(plt.imread(f), (dims, dims))  for f in files]
    imgs = [i for i in imgs if i.shape == (dims, dims, 3)]
    labels = np.array([[label, 1 - label] for i in imgs])
    data = np.array(imgs)

    return data, labels


def get_data(path_1, path_2):
    """
    input: path1 specifying data of the type we are trying to identify (1)
           path2 specifying data of the type we are not (0)
    output: the labeled data set
    """

    positive_data, positive_label = load_images(path_1, 1)
    negative_data, negative_label = load_images(path_2, 0)

    data = np.concatenate((positive_data, negative_data))
    labels = np.concatenate((positive_label, negative_label))

    return data, labels
