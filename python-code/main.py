import os
import numpy as np
import matplotlib.pyplot as plt
from calculate_tplans import calculate_tplans
import pixel_tplans
import utils
import cupy as cp
from tqdm import tqdm

"""
    N.B) For all images in the directories listed below:
        channel 0 --> Bassoon
        channel 1 --> PSD95
"""
BassoonFUS_PLKO = './jmdata_ot/composite/bassoon_fus/plko'
BassoonFUS_318 = './jmdata_ot/composite/bassoon_fus/318'

def extract_crops(directory):
    """
    Takes a directory and splits all the images it contains into crops
    Default size of the crops is 128x128

    Params:
    ---------
    directory (string): path to the directory which contains the images to split into crops

    Returns:
    all_crops_a, all_crops_b (list, list): lists of crops of images
    """
    tiff_files = [fname for fname in os.listdir(directory) if fname.endswith('.tif')]
    tiff_files = [os.path.join(directory, p) for p in tiff_files]
    all_crops_a, all_crops_b = [], []
    for img_file in tiff_files:
        crops_a, crops_b = utils.img_to_crops(img_file)
        all_crops_a += crops_a
        all_crops_b += crops_b
    assert len(all_crops_a) == len(all_crops_b)
    return (all_crops_a, all_crops_b)

def compute_OTC(crops, max_dist=30):
    """
    Computes the optimal transport curve considering distances going up to max_dist.
    One point on the transport curve is the fraction of the total mass transported given a maximum distance allowed for transportation
    Considering many such distances gives the full optimal transport curve

    Params:
    ----------
    crops (list): list of image crops for which to do the computation
    max_dist (int): maximum distance to consider when computing the OTC

    Returns:
    ----------
    otc_avg (array): Optimal transport curve average over the number of crops
    otc_std (array): Standard deviations of the OTC at each distance considered 
    """
    distances = cp.arange(0, 30, 1)
    (crops_a, crops_b) = crops
    ot_curve = cp.zeros((len(crops_a), distances.shape[0]))
    for i, (c_a, c_b) in enumerate(zip(crops_a, crops_b)):
        print(f"Processing crop {i + 1} of {len(crops_a)}")
        transport_plan, cost_matrix = pixel_tplans.calculate_tplans([c_a, c_b])
        otc_values = []
        for d in tqdm(distances, desc='Looping through distances'):
            transported_mass = utils.compute_transported_mass(transport_plan, cost_matrix, d)
            otc_values.append(transported_mass)
        otc_values = cp.array(otc_values)
        ot_curve[i] = otc_values
    otc_avg = cp.mean(ot_curve, axis=0)
    otc_std = cp.std(ot_curve, axis=0)
    return otc_avg, otc_std, distances





def main():
    # Extract crops
    crops = extract_crops(BassoonFUS_PLKO)
    # Compute OTC for all crops
    otc_avg, otc_std, distances = compute_OTC(crops)
    otc_avg, otc_std, distances = cp.asnumpy(otc_avg), cp.asnumpy(otc_std), cp.asnumpy(distances)

    crops_inf = extract_crops(BassoonFUS_318)
    otc_inf_avg, otc_inf_std, _ = compute_OTC(crops_inf)
    otc_inf_avg, otc_inf_std = cp.asnumpy(otc_inf_avg), cp.asnumpy(otc_inf_std)
    # Plot the results
    xtick_locs = [1, 5, 10, 15, 20, 25, 30]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    fig = plt.figure()
    plt.plot(distances, otc_avg, color='lightblue', label='PLKO')
    plt.fill_between(distances, otc_avg - otc_std, otc_avg + otc_std, facecolor='lightblue', alpha=0.5)
    plt.plot(distances, otc_inf_avg, color='lightcoral', label='shFUS-318')
    plt.fill_between(distances, otc_inf_avg - otc_inf_std, otc_inf_avg + otc_inf_std, facecolor='lightcoral', alpha=0.5)
    plt.xlabel('Distance (nm)', fontsize=14)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=14)
    plt.legend()
    plt.title('Bassoon - FUS', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('./results/temp.png')

if __name__ == "__main__":
    main()
