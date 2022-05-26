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
    img_a, img_b = imgs[0], imgs[1] # ( CAMKII, Actin)

    assert img_a.shape == img_b.shape

    img_a_pos = np.zeros((img_a.shape[0] * img_a.shape[1], 2))
    img_a_prod = np.zeros((img_a.shape[0] * img_b.shape[1],))
    img_b_prod = np.zeros((img_b.shape[0] * img_b.shape[1], ))
    
    print("Initializing positions and masses")
    index = 0
    for i in range(img_a.shape[0]):
        for j in range(img_a.shape[1]):
            img_a_pos[index][0] = i
            img_a_pos[index][1] = j
            img_a_prod[index] = img_a[i, j]
            img_b_prod[index] = img_b[i, j]
            index += 1
    img_b_pos = img_a_pos
    
      # normalize mass to be transported
    img_a_prod = img_a_prod / img_a_prod.sum()
    img_b_prod = img_b_prod / img_b_prod.sum()
    
    print("Computing cost matrix")
    cost_matrix = ot.dist(img_a_pos, img_b_pos, metric='euclidean')
    print("Computing transport plan")
    transport_plan = ot.emd(img_a_prod, img_b_prod, cost_matrix, numItermax=500000)
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0][0].imshow(img_a, cmap='hot')
    axs[0][0].set_title('CaMKII')
    axs[0][1].imshow(img_b, cmap='hot')
    axs[0][1].set_title('Actin')
    cmat = axs[1][0].imshow(cost_matrix, cmap='coolwarm')
    cbar = plt.colorbar(cmat, ax=axs[1][0], shrink=0.3)
    cbar.ax.set_xlabel('cost')
    axs[1][0].set_xlabel('Actin')
    axs[1][0].set_ylabel('CaMKII')
    cmat2 = axs[1][1].imshow(transport_plan, cmap='coolwarm')
    cbar = plt.colorbar(cmat2, ax=axs[1], shrink=0.3)
    cbar.ax.set_xlabel('transport plan')
    plt.tight_layout()
    fig.savefig('temp.png')
    plt.close(fig)
    """
    return transport_plan, cost_matrix
