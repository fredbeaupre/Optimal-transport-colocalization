import numpy as np
from utils import compute_OTC_v2


def main():
    mat = np.zeros((5, 3))
    mat[0] = [1, 2, 3]
    mat[1] = [6, 6, 6]
    mat[2] = [4, 4, 4]
    mat[3] = [0, 6, 0]
    mat[4] = [7, 7, 7]
    tplan = np.zeros((5, 3))
    tplan[0] = [1, 2, 3]
    tplan[1] = [1000, 1000, 1000]
    tplan[2] = [1, 2, 3]
    tplan[3] = [1, 2, 3]
    tplan[4] = [1, 2, 3]
    boolmat = compute_OTC_v2(tplan, mat, 5)
    print(boolmat)


if __name__ == "__main__":
    main()
