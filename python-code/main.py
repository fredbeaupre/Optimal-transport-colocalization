import os
import numpy as np
import matplotlib.pyplot as plt
from pytz import common_timezones
from calculate_tplans import calculate_tplans
from utils import normalize_mass, compute_OTC

"""
    N.B) For all images in the directories listed below:
        channel 0 --> Bassoon
        channel 1 --> PSD95
"""
BLOCK_DIR = "../../dataset_creator/theresa-dataset/DATA_SYNPROT_BLOCKGLUGLY/Block"
GLUGLY_DIR = "../../dataset_creator/theresa-dataset/DATA_SYNPROT_BLOCKGLUGLY/GluGLY"


def main():
    common_dist = np.linspace(0, 1000, 1000)
    otc_avg = None
    otc_std = None
    otc_dist = None
    gg_avg = None
    gg_std = None
    gg_dist = None
    for dir_idx, direc in enumerate([BLOCK_DIR, GLUGLY_DIR]):
        tiff_files = os.listdir(direc)
        tiff_files = [os.path.join(direc, p) for p in tiff_files]
        N = 1000
        num_files = len(tiff_files)
        block_otc = np.zeros((num_files, N))
        for i in range(num_files):
            print("\nProcessing file {} of {}".format(i + 1, num_files))
            transport_plan, cost_matrix = calculate_tplans(imgs=tiff_files[i])

            # distances = np.linspace(np.min(cost_matrix),
            #                         1000, num=N)
            otc_values = []
            for _, t in enumerate(common_dist):
                transported_mass = compute_OTC(transport_plan, cost_matrix, t)
                otc_values.append(transported_mass)
            otc_values = np.array(otc_values)
            block_otc[i] = otc_values

        block_avg = np.mean(block_otc, axis=0)
        block_std = np.std(block_otc, axis=0)
        # argmax = np.where(block_avg > 0.99)[0][0]
        # distances_updated = distances[:argmax]
        # block_avg_updated = block_avg[:argmax]
        # block_std_updated = block_std[:argmax]
        if dir_idx == 0:
            otc_avg = block_avg
            # otc_dist = distances_updated
            otc_std = block_std
        else:
            gg_avg = block_avg
            # gg_dist = distances_updated
            gg_std = block_std

    fig = plt.figure()
    plt.plot(common_dist, otc_avg,
             color='lightblue', label='Block')
    plt.fill_between(common_dist, otc_avg - otc_std,
                     otc_avg+otc_std, facecolor='lightblue', alpha=0.5)
    plt.plot(common_dist, gg_avg, color='lightcoral', label='GluGly')
    plt.fill_between(common_dist, gg_avg - gg_std,
                     gg_avg + gg_std, facecolor='lightcoral', alpha=0.5)
    plt.xlabel('Distance', fontsize=16)
    plt.xticks(rotation=-30)
    plt.ylabel('OTC', fontsize=16)
    plt.legend()
    plt.title('Bassoon - PSD95 OTC', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('bassoon_psd95_OTC_1500x1500.png')


if __name__ == "__main__":
    main()
