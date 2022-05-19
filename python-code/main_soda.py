import os
import numpy as np
import matplotlib.pyplot as plt
from soda_tplans import soda_tplans
from utils import normalize_mass, compute_OTC, compute_OTC_v2
from tqdm import tqdm

"""
    N.B) For all images in the directories listed below:
        channel 0 --> Bassoon
        channel 1 --> PSD95
"""
BLOCK_DIR = "../../dataset_creator/theresa-dataset/DATA_SYNPROT_BLOCKGLUGLY/Block"
GLUGLY_DIR = "../../dataset_creator/theresa-dataset/DATA_SYNPROT_BLOCKGLUGLY/GluGLY"

BassoonFUS_PLKO = './jmdata_ot/composite/bassoon_fus/plko'
BassoonFUS_318 = './jmdata_ot/composite/bassoon_fus/318'

BassoonPSD_PLKO = './jmdata_ot/composite/bassoon_psd/plko'
BassoonPSD_318 = './jmdata_ot/composite/bassoon_psd/318'


def main():
    N = 1500
    common_dist = np.linspace(0, N, N)
    block_avg = None
    block_std = None
    block_dist = None
    gg_avg = None
    gg_std = None
    gg_dist = None
    for dir_idx, direc in enumerate([BassoonFUS_PLKO, BassoonFUS_318]):
        tiff_files = [fname for fname in os.listdir(
            direc) if fname.endswith('.tif')]
        tiff_files = [os.path.join(direc, p) for p in tiff_files]
        num_files = len(tiff_files)
        ot_curve = np.zeros((num_files, N))
        for i in range(num_files):
            print("\nProcessing image {} of {}".format(i + 1, num_files))
            transport_plan, cost_matrix = soda_tplans(imgs=tiff_files[i])
            otc_values = []
            for dist in tqdm(common_dist):
                transported_mass = compute_OTC_v2(
                    transport_plan, cost_matrix, dist)
                otc_values.append(transported_mass)
            otc_values = np.array(otc_values)
            ot_curve[i] = otc_values
        ot_avg = np.mean(ot_curve, axis=0)
        ot_std = np.std(ot_curve, axis=0)
        if dir_idx == 0:
            block_avg = ot_avg
            block_std = ot_std
        else:
            gg_avg = ot_avg
            gg_std = ot_std

    xtick_locs = [200, 600, 1000, 1400]
    xtick_labels = [str(item * 15) for item in xtick_locs]
    fig = plt.figure()
    plt.plot(common_dist, block_avg,
             color='lightblue', label='PLKO')
    plt.fill_between(common_dist, block_avg - block_std,
                     block_avg + block_std, facecolor='lightblue', alpha=0.5)
    plt.plot(common_dist, gg_avg, color='lightcoral', label='shFus318')
    plt.fill_between(common_dist, gg_avg - gg_std,
                     gg_avg + gg_std, facecolor='lightcoral', alpha=0.5)
    plt.xlabel('Distance (nm)', fontsize=14)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=14)
    plt.legend()
    plt.title('Bassoon - FUS OTC', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('bassoon_psd_jmichel_OTC_soda_zoomed.png')


if __name__ == "__main__":
    main()