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
    ctrl_avg = None
    ctrl_std = None
    disease_avg = None
    disease_std = None
    for dir_idx, direc in enumerate([BassoonFUS_PLKO, BassoonFUS_318]):
        tiff_files = [fname for fname in os.listdir(
            direc) if fname.endswith('.tif')]
        tiff_files = [os.path.join(direc, p) for p in tiff_files]
        num_files = len(tiff_files)
        ot_curve = np.zeros((num_files, N))
        for i in range(num_files):
            print("\nProcessing image {} of {}".format(i + 1, num_files))
            # compute cost matrix and transport plan
            transport_plan, cost_matrix = soda_tplans(imgs=tiff_files[i])
            otc_values = []
            # computes fraction of mass being transported for a range of distances
            for dist in tqdm(common_dist):
                transported_mass = compute_OTC_v2(
                    transport_plan, cost_matrix, dist)
                otc_values.append(transported_mass)
            otc_values = np.array(otc_values)
            ot_curve[i] = otc_values
        ot_avg = np.mean(ot_curve, axis=0)
        ot_std = np.std(ot_curve, axis=0)
        if dir_idx == 0:
            ctrl_avg = ot_avg
            ctrl_std = ot_std
        else:
            disease_avg = ot_avg
            disease_std = ot_std

    # plot the OT curves
    xtick_locs = [200, 600, 1000, 1400]
    # convert distance from pixels to nm, assuming pixel size = 15
    xtick_labels = [str(item * 15) for item in xtick_locs]
    fig = plt.figure()
    plt.plot(common_dist, ctrl_avg,
             color='lightblue', label='PLKO')
    plt.fill_between(common_dist, ctrl_avg - ctrl_std,
                     ctrl_avg + ctrl_std, facecolor='lightblue', alpha=0.5)
    plt.plot(common_dist, disease_avg, color='lightcoral', label='shFus318')
    plt.fill_between(common_dist, disease_avg - disease_std,
                     disease_avg + disease_std, facecolor='lightcoral', alpha=0.5)
    plt.xlabel('Distance (nm)', fontsize=14)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=14)
    plt.legend()
    plt.title('Bassoon - FUS OTC', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('./figures/bassoon_psd_jmichel_OTC_soda_zoomed.png')


if __name__ == "__main__":
    main()
