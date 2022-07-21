import numpy as np
import tifffile
import ot
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from utils import *
import time


def calculate_tplans(imgs):

    # reading images
    img_a = tifffile.imread(imgs)[0]
    img_b = tifffile.imread(imgs)[1]
    # denoising
    img_a = img_a * (img_a > np.percentile(img_a, 100 * 0.05))
    img_b = img_b * (img_b > np.percentile(img_b, 100 * 0.05))
    # normalizing
    img_a_og = img_a / img_a.sum()
    img_b_og = img_b / img_b.sum()

    # taking 1024x1024 random crop
    img_a, img_b = random_crop(img_a_og, img_b_og, 1000, 1000)
    num_pixels_a = np.count_nonzero(img_a)
    num_pixels_b = np.count_nonzero(img_b)
    percentage_a = num_pixels_a / \
        (img_a.shape[0] * img_a.shape[1])
    percentage_b = num_pixels_b / \
        (img_b.shape[0] * img_b.shape[1])
    good_crop = True if percentage_a > 0.40 and percentage_b > 0.35 else False
    # print(round(percentage_a, 3), round(percentage_b, 3))
    crops_tried = 0
    # keep cropping until we have a decent crop, i.e., one with enough structures in it
    while not good_crop:
        crops_tried += 1
        # Get segmentation masks from imgs using otsu's method
        # check non-zero structures in the binary masks --> to check if good crop
        num_pixels_a = np.count_nonzero(img_a)
        num_pixels_b = np.count_nonzero(img_b)
        percentage_a = num_pixels_a / \
            (img_a.shape[0] * img_a.shape[1])
        percentage_b = num_pixels_b / \
            (img_b.shape[0] * img_b.shape[1])
        # print(round(percentage_a, 3), round(percentage_b, 3))
        good_crop = True if percentage_a > 0.40 and percentage_b > 0.35 else False
        img_a, img_b = random_crop(img_a_og, img_b_og, 1000, 1000)

    print("Tried {} crops before finding a good one".format(crops_tried))

    # Generate binary mask
    img_a_binary = img_binary_mask(img_a)
    img_b_binary = img_binary_mask(img_b)
    # # Closing
    # bw_a = closing(img_a_binary, square(3))
    # bw_b = closing(img_b_binary, square(3))
    # # remove artifacts connected to image border
    # bw_a_cleared = clear_border(bw_a)
    # bw_b_cleared = clear_border(bw_b)

    # Extract regions from the binary masks
    img_b_clusters, nd_components_b = clusterize(img_b_binary)
    img_a_clusters, nd_components_a = clusterize(img_a_binary)
    img_a_rprops = regionprops(img_a_clusters)
    img_b_rprops = regionprops(img_b_clusters)

    # Removing small regions
    img_a_rprops, nd_components_a = remove_small_regions(img_a_rprops)
    img_b_rprops, nd_components_b = remove_small_regions(img_b_rprops)

    # Initializing and populating data necessary for the optimal transport problem
    img_a_pos = np.zeros((nd_components_a, 2))
    img_a_prod = np.zeros((nd_components_a,))
    img_b_pos = np.zeros((nd_components_b, 2))
    img_b_prod = np.zeros((nd_components_b,))
    for i, r in enumerate(img_a_rprops):
        img_a_pos[i][0] = r.centroid[0]
        img_a_pos[i][1] = r.centroid[1]
        min_row, min_col, max_row, max_col = r.bbox
        roi = img_a[min_row:max_row, min_col:max_col]
        mass = np.mean(roi)
        img_a_prod[i] = mass

    for i, r in enumerate(img_b_rprops):
        img_b_pos[i][0] = r.centroid[0]
        img_b_pos[i][1] = r.centroid[1]
        min_row, min_col, max_row, max_col = r.bbox
        roi = img_b[min_row:max_row, min_col:max_col]
        mass = np.mean(roi)
        img_b_prod[i] = mass

    # Normalize the mass arrays
    img_a_prod = img_a_prod / img_a_prod.sum()
    img_b_prod = img_b_prod / img_b_prod.sum()

    # Computing the cost matrix
    cost_matrix = ot.dist(img_a_pos, img_b_pos, metric='euclidean')
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img_a_clusters > 0, cmap='gray')
    axs[1].imshow(img_b_clusters > 0, cmap="gray")
    plt.tight_layout()
    fig.savefig('synprot_channels.png', bbox_inches='tight')
    plt.close()

    # Compute transport plan
    start = time.time()
    transport_plan = ot.emd(img_a_prod, img_b_prod, cost_matrix)
    transport_plan = normalize_mass(transport_plan)
    time_emd = time.time() - start
    print("Compute time for transport plan: {}".format(time_emd))

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    im1 = axs[0].imshow(cost_matrix, cmap='coolwarm')
    cbar = plt.colorbar(im1, ax=axs[0], shrink=0.3)
    cbar.ax.set_xlabel('cost')
    axs[0].set_xlabel('PSD-95')
    axs[0].set_ylabel('Bassoon')
    im2 = axs[1].imshow(transport_plan, cmap='coolwarm')
    cbar = plt.colorbar(im2, ax=axs[1], shrink=0.3)
    cbar.ax.set_xlabel('transport plan')
    axs[1].set_xlabel('PSD-95')
    axs[1].set_ylabel('Bassoon')
    plt.tight_layout()
    fig.savefig('synprot_transport_plan.png', bbox_inches='tight')
    plt.close()
    return transport_plan, cost_matrix
