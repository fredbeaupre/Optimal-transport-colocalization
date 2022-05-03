import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import ot
import time
from utils import normalize_mass, compute_OTC
plt.style.use('dark_background')


def load_data():
    return np.load('./manhattan.npz')


def plot_data(data):
    bakery_pos = data['bakery_pos']
    bakery_prod = data['bakery_prod']
    cafe_pos = data['cafe_pos']
    cafe_prod = data['cafe_prod']
    Imap = data['Imap']

    pl.figure(1, (7, 6))
    pl.clf()
    pl.imshow(Imap, interpolation='bilinear')  # plot the map
    pl.scatter(bakery_pos[:, 0], bakery_pos[:, 1],
               s=bakery_prod, c='r', ec='k', label='Bakeries')
    pl.scatter(cafe_pos[:, 0], cafe_pos[:, 1],
               s=cafe_prod, c='b', ec='k', label='Cafés')
    pl.legend()
    pl.title('Manhattan Bakeries and Cafés')
    pl.show()


def main():
    data = load_data()
    # plot_data(data)
    bakery_pos = data['bakery_pos']
    bakery_prod = data['bakery_prod']
    cafe_pos = data['cafe_pos']
    cafe_prod = data['cafe_prod']
    print(bakery_pos)
    print('\n')
    print(cafe_pos)
    exit()

    print("Bakery production: {}".format(bakery_prod))
    print("Total: {}".format(bakery_prod.sum()))
    print("Cafe sales: {}".format(cafe_prod))
    print("Total: {}\n".format(cafe_prod.sum()))
    Imap = data['Imap']
    # compute the distance between points in x and points in y, i.e, the cost matrix
    cost_matrix = ot.dist(bakery_pos, cafe_pos)
    labels = [str(i) for i in range(len(bakery_prod))]

    # # Uncomment this to visualize the cost matrix
    # f = pl.figure(2, (14, 7))
    # pl.clf()
    # pl.subplot(121)
    # pl.imshow(Imap, interpolation='bilinear')  # plot the map
    # for i in range(len(cafe_pos)):
    #     pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b',
    #             fontsize=14, fontweight='bold', ha='center', va='center')
    # for i in range(len(bakery_pos)):
    #     pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r',
    #             fontsize=14, fontweight='bold', ha='center', va='center')
    # pl.title('Manhattan Bakeries and Cafés')

    # ax = pl.subplot(122)
    # ax.xaxis.label.set_color('lightblue')
    # ax.yaxis.label.set_color('lightcoral')
    # im = pl.imshow(cost_matrix, cmap="coolwarm")
    # pl.title('Cost matrix')
    # cbar = pl.colorbar(im, ax=ax, shrink=0.5, use_gridspec=True)
    # cbar.ax.set_ylabel("cost", rotation=-90, va="bottom")
    # pl.xlabel('Cafés')
    # pl.ylabel('Bakeries')
    # pl.tight_layout()
    # pl.show()
    # f.savefig('manhattan_cost_matrix.png')
    # exit()

    start = time.time()
    # Calculating transport plan
    transport_plan = ot.emd(bakery_prod, cafe_prod, cost_matrix)
    time_emd = time.time() - start
    print("Compute time for the transport plan: {}".format(time_emd))

    # # Uncomment this to visualize the transport plan
    # f = pl.figure(3, (14, 7))
    # pl.clf()
    # pl.subplot(121)
    # pl.imshow(Imap, interpolation='bilinear')  # plot the map
    # for i in range(len(bakery_pos)):
    #     for j in range(len(cafe_pos)):
    #         pl.plot([bakery_pos[i, 0], cafe_pos[j, 0]], [bakery_pos[i, 1], cafe_pos[j, 1]],
    #                 '-k', lw=3. * transport_plan[i, j] / transport_plan.max())
    # for i in range(len(cafe_pos)):
    #     pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b', fontsize=14,
    #             fontweight='bold', ha='center', va='center')
    # for i in range(len(bakery_pos)):
    #     pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r', fontsize=14,
    #             fontweight='bold', ha='center', va='center')
    # pl.title('Manhattan Bakeries and Cafés')

    # ax = pl.subplot(122)
    # im = pl.imshow(transport_plan)
    # for i in range(len(bakery_prod)):
    #     for j in range(len(cafe_prod)):
    #         text = ax.text(j, i, '{0:g}'.format(transport_plan[i, j]),
    #                        ha="center", va="center", color="w")
    # pl.title('Transport matrix')

    # pl.xlabel('Cafés')
    # pl.ylabel('Bakeries')
    # pl.tight_layout()
    # pl.show()
    # f.savefig('manhattan_transport_plan.png')
    # exit()

    distances = np.linspace(np.min(cost_matrix), np.max(cost_matrix), num=1000)
    transport_plan = normalize_mass(transport_plan)
    otc_values = []
    for t in distances:
        transported_mass = compute_OTC(transport_plan, cost_matrix, t)
        otc_values.append(transported_mass)
    fig = plt.figure()
    plt.plot(distances, otc_values, color='lightblue', label='Bakeries/Cafes')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('OTC', fontsize=16)
    plt.legend()
    plt.title('OTC for Manhattan Cafes and Bakeries', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    fig.savefig('manhattan_otc.png')


if __name__ == "__main__":
    main()
