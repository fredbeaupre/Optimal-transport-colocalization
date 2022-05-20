import os
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_OTC_v2
from split_crops import img_to_crops
from rings_tplans import calculate_tplans
from tqdm import tqdm
from soda_tplans import soda_tplans


TEST_IMG = './EXP9-GFP-compositeSTED-4-couleurs/9-GFP-LAM18-GFP-TAUS460L-BCaMKIIA594-PH635_cs1n6_CompositeSTED.tiff'


def main():
    N = 128
    common_dist = np.linspace(0, N, N)
    crops_a, crops_b = img_to_crops(TEST_IMG)
    calculate_tplans([crops_a[0], crops_b[0]])
    exit()
    ot_curve = np.zeros((len(crops_a), N))
    for i, (crop_a, crop_b) in enumerate(zip(crops_a, crops_b)):
        print("Processing image {} of {}".format(i + 1, len(crops_a)))
        transport_plan, cost_matrix = soda_tplans(
            imgs=[crop_a, crop_b], use_crops=True)
        otc_values = []
        for dist in tqdm(common_dist):
            transported_mass = compute_OTC_v2(
                transport_plan, cost_matrix, dist)
            otc_values.append(transported_mass)
        otc_values = np.array(otc_values)
        ot_curve[i] = otc_values
    ot_avg = np.mean(ot_curve, axis=0)
    ot_std = np.std(ot_curve, axis=0)
    # xtick_locs = np.linspace(0, 128, 20)
    # xitck_labels = [str(item * 15) for item in xtick_locs]
    fig = plt.figure()
    plt.plot(common_dist, ot_avg, color='lightblue')
    plt.fill_between(common_dist, ot_avg - ot_std, ot_avg +
                     ot_std, facecolor='lightblue', alpha=0.5)
    plt.xlabel('Distance (pixels)', fontsize=14)
    plt.ylabel('OTC', fontsize=14)
    plt.title('OTC: CaMKII - Actin', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('./figures/CaMKII_Actin_FLAVIE/camkii_actin_otcs.png')


if __name__ == "__main__":
    main()
