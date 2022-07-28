import numpy as np
from scipy.ndimage import measurements
from skimage.filters import threshold_otsu
import random
import tifffile
import cupy as cp
from scipy.stats import pearsonr


def remove_small_regions(rprops):
    new_rprops = []
    for i, r in enumerate(rprops):
        min_row, min_col, max_row, max_col = r.bbox
        height = max_row - min_row
        width = max_col - min_col
        if height > 1 and width > 1:
            new_rprops.append(r)
    return new_rprops, len(new_rprops)


def random_crop(img_a, img_b, new_w, new_h):
    assert img_a.shape == img_b.shape
    height = img_a.shape[0]
    width = img_a.shape[1]
    x = random.randint(0, width - new_w)
    y = random.randint(0, height - new_h)
    new_a = img_a[y:y + new_h, x:x + new_w]
    new_b = img_b[y:y + new_h, x:x + new_w]
    return new_a, new_b


def img_binary_mask(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary


def clusterize(img):
    img = img > 0
    dims = len(img.shape)
    structure = np.ones(np.ones(dims, int) * 3)
    labeled, ncomponents = measurements.label(img, structure)
    return labeled, ncomponents


def im2mat(img):
    """Converts an image to a matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], 1))


def mat2im(X, shape):
    """Converts a matrix back to an image"""
    return X.reshape(shape)


def normalize_mass(transport_plan):
    total_mass = np.sum(transport_plan)
    return transport_plan / total_mass


def compute_OTC(transport_plan, cost_matrix, dist):
    """
    This function does the same thing as `compute_transported_mass` below but in a much slower way
    """
    assert transport_plan.shape == cost_matrix.shape
    transported_mass = 0
    row, col = transport_plan.shape[0], transport_plan.shape[1]
    for i in range(row):
        for j in range(col):
            if transport_plan[i][j] != 0 and cost_matrix[i][j] <= dist:
                transported_mass += transport_plan[i][j]
    return transported_mass


def compute_transported_mass(transport_plan, cost_matrix, dist):
    """
    Computes the amount of transported mass that can be transported for a given maximum distance of transport

    Params:
    --------
    transport plan (WxH matrix): optimal transport plan between two distributions
    cost matrix (WxH matrix): cost matrix of the cost matrix between each pair of elements in the two distributions
    dist (int): maximum distance in pixels allowed for a transport

    Returns:
    ---------
    Fraction of the total mass transported
    """
    gt_dist = cost_matrix <= dist
    return np.sum(transport_plan[gt_dist == True])


def random_distributions(size=128):
    """
    Creates two images both with random spatial distributions

    Params:
    --------
    size (int): size of the images
    """
    img_a = np.zeros((size, size))
    img_b = np.zeros((size, size))
    num_regions_a = np.random.randint(5, 20)
    num_regions_b = np.random.randint(5, 20)
    for _ in range(num_regions_a):
        x_size = np.random.randint(2, 10)
        y_size = np.random.randint(2, 10)
        x = np.random.randint(0, size - x_size)
        y = np.random.randint(0, size - y_size)
        for i in range(x, x + x_size):
            for j in range(y, y + y_size):
                img_a[i, j] = np.random.uniform()
    for _ in range(num_regions_b):
        x_size = np.random.randint(2, 10)
        y_size = np.random.randint(2, 10)
        x = np.random.randint(0, size - x_size)
        y = np.random.randint(0, size - y_size)
        for i in range(x, x + x_size):
            for j in range(y, y + y_size):
                img_b[i, j] = np.random.uniform()
    return img_a, img_b


def highly_colocalized_distributions(size=128):
    """
    Builds two images which have high colocalization in their structures

    Params:
    --------
    size (int): size of the images
    """
    img_a = np.zeros((size, size))
    img_b = np.zeros(img_a.shape)
    num_regions = np.random.randint(5, 20)
    for _ in range(num_regions):
        r_size = np.random.randint(2, 20)
        x_a = np.random.randint(0, size - 2 * r_size)
        y_a = np.random.randint(0, size - 2 * r_size)
        x_b = int(np.floor(np.random.normal(loc=x_a, scale=2.0)))
        y_b = int(np.floor(np.random.normal(loc=y_a, scale=2.0)))
        for i in range(x_a, x_a + r_size):
            for j in range(y_a, y_a + r_size):
                img_a[i, j] = random.random()
        for i in range(x_b, x_b + r_size):
            for j in range(y_b, y_b + r_size):
                img_b[i, j] = random.random()
    pearson, _ = pearsonr(img_a.flatten(), img_b.flatten())
    return img_a, img_b, pearson


def build_colocalized_dataset():
    """
    Builds a dataset of crops in which the regions are highly colocalized
    """
    crops_a, crops_b = [], []
    pearson_coeffs = []
    for i in range(400):
        img_a, img_b, pearson = highly_colocalized_distributions()
        pearson_coeffs.append(pearson)
        crops_a.append(img_a)
        crops_b.append(img_b)

    return crops_a, crops_b, np.array(pearson_coeffs)


def build_random_dist_dataset():
    """
    Builds a dataset of crops in which the regions follow a random spatial distribution
    """
    crops_a, crops_b = [], []
    for i in range(400):
        img_a, img_b = random_distributions()
        crops_a.append(img_a)
        crops_b.append(img_b)
    return crops_a, crops_b


def img_to_crops(img, size=128, step=96):
    """
    Takes a large image and extracts multiple (can be overlapping) crops.

    Returns:
    ------------
    List of extracted crops
    """
    img_a = tifffile.imread(img)[0]  # Bassoon / VGLUT1 / CaMKII
    img_b = tifffile.imread(img)[1]  # PSD-95 / Actin
    x, y = img_a.shape
    assert x == img_b.shape[0]
    assert y == img_b.shape[1]
    start_xs = cp.arange(0, x - size, step)
    start_xs = [int(cp.floor(x)) for x in start_xs]
    start_ys = cp.arange(0, y - size, step)
    start_ys = [int(cp.floor(y)) for y in start_ys]
    crops_a = []
    crops_b = []
    img_a_mean = cp.mean(img_a)
    img_b_mean = cp.mean(img_b)
    for s_x in start_xs:
        for s_y in start_ys:
            crop_a = img_a[s_x:s_x + size, s_y:s_y + size]
            crop_b = img_b[s_x:s_x + size, s_y:s_y + size]
            crop_a_mean = cp.mean(crop_a)
            crop_b_mean = cp.mean(crop_b)
            if (crop_a_mean >= img_a_mean) and (crop_b_mean >= img_b_mean):
                crops_a.append(crop_a)
                crops_b.append(crop_b)
    return crops_a, crops_b
