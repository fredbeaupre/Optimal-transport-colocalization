import numpy as np
from scipy.ndimage import measurements
from skimage.filters import threshold_otsu
import random
import tifffile
import cupy as cp


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
    new_a = img_a[y:y+new_h, x:x+new_w]
    new_b = img_b[y:y+new_h, x:x+new_w]
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
    assert transport_plan.shape == cost_matrix.shape
    transported_mass = 0
    row, col = transport_plan.shape[0], transport_plan.shape[1]
    for i in range(row):
        for j in range(col):
            if transport_plan[i][j] != 0 and cost_matrix[i][j] <= dist:
                transported_mass += transport_plan[i][j]
    return transported_mass


def compute_transported_mass(transport_plan, cost_matrix, dist):
    gt_dist = cost_matrix <= dist
    return np.sum(transport_plan[gt_dist == True])

def img_to_crops(img, size=128):
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
    start_xs = cp.arange(0, x - size, size)
    start_xs = [int(cp.floor(x)) for x in start_xs]
    start_ys = cp.arange(0, y - size, size)
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