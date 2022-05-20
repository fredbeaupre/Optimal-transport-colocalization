import matplotlib.pyplot as plt
from utils import *
from skimage import filters, morphology
from skimage.measure import label, regionprops
import ot
import tifffile
import os
import numpy as np
import time
import sys
from split_crops import img_to_crops
plt.style.use('dark_background')


def calculate_tplans(imgs):
    img_a, img_b = imgs[0], imgs[1]
    # normalize mass to be transported
    img_a_prod = img_a / img_a.sum()
    img_b_prod = img_b / img_b.sum()

    assert img_a.shape == img_b.shape

    img_a_pos = np.zeros((img_a.shape[0] * img_a.shape[1], 2))
    img_a_prod = np.zeros((img_a.shape[0] * img_b.shape[1],))
    img_b_prod = np.zeros((img_b.shape[0] * img_b.shape[1], ))

    index = 0
    for i in range(img_a.shape[0]):
        print("{}/{}".format(index, img_a.shape[0] * img_a.shape[1]))
        for j in range(img_a.shape[1]):
            img_a_pos[index][0] = i
            img_a_pos[index][1] = j
            img_a_prod[index] = img_a[i, j]
            img_b_prod[index] = img_b[i, j]
            index += 1
    img_b_pos = img_a_pos

    cost_matrix = ot.dist(img_a_pos, img_b_pos, metric='euclidean')
    plt.imshow(cost_matrix, cmap='coolwarm')
    plt.show()
