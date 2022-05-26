import numpy as np
import tifffile


def img_to_crops(img, size=128):
    """
    Takes a large image and extracts multiple (can be overlapping) crops.

    Returns:
    ------------
    List of extracted crops
    """
    img_a = tifffile.imread(img)[2]  # CaMKII
    img_b = tifffile.imread(img)[3]  # Actin
    x, y = img_a.shape
    assert x == img_b.shape[0]
    assert y == img_b.shape[1]
    start_xs = np.arange(0, x - size, 64)
    start_xs = [int(np.floor(x)) for x in start_xs]
    start_ys = np.arange(0, y - size, 64)
    start_ys = [int(np.floor(y)) for y in start_ys]
    crops_a = []
    crops_b = []
    img_a_mean = np.mean(img_a)
    img_b_mean = np.mean(img_b)
    for s_x in start_xs:
        for s_y in start_ys:
            crop_a = img_a[s_x:s_x + size, s_y:s_y + size]
            crop_b = img_b[s_x:s_x + size, s_y:s_y + size]
            crop_a_mean = np.mean(crop_a)
            crop_b_mean = np.mean(crop_b)
            if (crop_a_mean >= img_a_mean) and (crop_b_mean >= img_b_mean):
                crops_a.append(crop_a)
                crops_b.append(crop_b)
    return crops_a, crops_b
