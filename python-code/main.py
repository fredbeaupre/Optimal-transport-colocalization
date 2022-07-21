import os
import numpy as np
import matplotlib.pyplot as plt
from calculate_tplans_old import calculate_tplans
import pixel_tplans
import utils
import cupy as cp
from tqdm import tqdm
import scipy.stats

"""
 BassoonFUS: Bassoon in channel 0, FUS in channel 1
 BassoonPSD: Bassoon in channel 0, PSD95 in channel 1
 ActinCaMKII: 

"""
BassoonFUS_PLKO = './jmdata_ot/composite/bassoon_fus/plko'
BassoonFUS_318 = './jmdata_ot/composite/bassoon_fus/318'
BassoonPSD_PLKO = './jmdata_ot/composite/bassoon_psd/plko'
BassoonPSD_318 = './jmdata_ot/composite/bassoon_psd/318'

DATASET = './path/to/dataset'
OUTPUT = './results/<figure-name>'


def compute_confidence_interval(averages, stds):
    """
    Computes the 95% confidence interval values
    Note that the Student's t distribution is used to compute t_crit.

    Params:
    --------
    averages (array): array of OTC values for different maximum transport distances
    stds (array): array of standard deviations on the OTC values for different maximum transport distances

    Returns:
    ---------
    95% confidence interval values (lower and upper bounds)
    """
    dof = averages.shape[0] - 1
    confidence = 0.95
    t_crit = np.abs(scipy.stats.t.ppf((1 - confidence) / 2, dof))
    confidence = stds * t_crit / np.sqrt(averages.shape[0])
    return confidence


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
    tiff_files = [fname for fname in os.listdir(
        directory) if fname.endswith('.tif')]
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
    distances = cp.arange(0, 20, 1)
    (crops_a, crops_b) = crops
    num_samples = len(crops_a)
    ot_curve = cp.zeros((len(crops_a), distances.shape[0]))
    for i, (c_a, c_b) in enumerate(zip(crops_a, crops_b)):
        print(f"Processing crop {i + 1} of {len(crops_a)}")
        transport_plan, cost_matrix = pixel_tplans.calculate_tplans([c_a, c_b])
        otc_values = []
        for d in tqdm(distances, desc='Looping through distances'):
            transported_mass = utils.compute_transported_mass(
                transport_plan, cost_matrix, d)
            otc_values.append(transported_mass)
        otc_values = cp.array(otc_values)
        ot_curve[i] = otc_values
    otc_avg = cp.mean(ot_curve, axis=0)
    otc_std = cp.std(ot_curve, axis=0)
    confidence = compute_confidence_interval(otc_avg, otc_std)
    return otc_avg, otc_std, confidence, distances


def main():
    # Extract crops
    crops = extract_crops(BassoonFUS_PLKO)
    # Compute OTC for all crops
    otc_avg, otc_std, confidence, distances = compute_OTC(crops)
    otc_avg, otc_std, confidence, distances = cp.asnumpy(otc_avg), cp.asnumpy(
        otc_std), cp.asnumpy(confidence), cp.asnumpy(distances)

    crops_inf = extract_crops(BassoonFUS_318)
    otc_inf_avg, otc_inf_std, confidence_inf, _ = compute_OTC(crops_inf)
    otc_inf_avg, otc_inf_std, confidence_inf = cp.asnumpy(
        otc_inf_avg), cp.asnumpy(otc_inf_std), cp.asnumpy(confidence_inf)

    crops_random = utils.build_random_dist_dataset()
    otc_random_avg, otc_random_std, confidence_random, _ = compute_OTC(
        crops_random)
    otc_random_avg, otc_random_std, confidence_random = cp.asnumpy(
        otc_random_avg), cp.asnumpy(otc_random_std), cp.asnumpy(confidence_random)

    crops_loc = utils.build_colocalized_dataset()
    otc_loc_avg, otc_loc_std, confidence_loc, _ = compute_OTC(crops_loc)
    otc_loc_avg, otc_loc_std, confidence_loc = cp.asnumpy(
        otc_loc_avg), cp.asnumpy(otc_loc_std), cp.asnumpy(confidence_loc)

    # Plot the results
    xtick_locs = [1, 5, 10, 15, 20]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    np.savez('./results/BassoonFUS/plko', otc=otc_avg,
             distances=distances, confidence=confidence)
    np.savez('./results/BassoonFUS/shfus', otc=otc_inf_avg,
             distances=distances, confidence=confidence_inf)
    np.savez('./results/BassoonFUS/random', otc=otc_random_avg,
             distances=distances, confidence=confidence_random)
    np.savez('./results/BassoonFUS/random', otc=otc_loc_avg,
             distances=distances, confidence=confidence_loc)
    fig = plt.figure()
    # PLKO
    plt.plot(distances, otc_avg, color='lightblue', label='PLKO')
    plt.fill_between(distances, otc_avg - confidence, otc_avg +
                     confidence, facecolor='lightblue', alpha=0.5)
    # Disease
    plt.plot(distances, otc_inf_avg, color='lightcoral', label='shFUS-318')
    plt.fill_between(distances, otc_inf_avg - confidence_inf,
                     otc_inf_avg + confidence_inf, facecolor='lightcoral', alpha=0.5)
    # Random
    plt.plot(distances, otc_random_avg, color='limegreen', label='Random')
    plt.fill_between(distances, otc_random_avg - confidence_random,
                     otc_random_avg + confidence_random, facecolor='limegreen', alpha=0.5)

    # Higly colocalized
    plt.plot(distances, otc_loc_avg, color='mediumorchid',
             label='Highly colocalized')
    plt.fill_between(distances, otc_loc_avg - confidence_loc,
                     otc_loc_avg + confidence_loc, facecolor='mediumorchid', alpha=0.5)
    plt.xlabel('Distance (nm)', fontsize=14)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=14)
    plt.legend()
    plt.title('Bassoon - FUS', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig(OUTPUT,
                transparent=True)


if __name__ == "__main__":
    main()
