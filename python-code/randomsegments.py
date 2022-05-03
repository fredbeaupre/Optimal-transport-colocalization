import numpy as np


def randomsegments(img_1, img_2, sample_size=10, segment_len=200, segment_width=200, rel_mass=0.8):
    dims = img_1.shape

    truncated_len = dims[0] - segment_len + 1
    truncated_width = dims[1] - segment_len

    segments_1 = []
    segments_2 = []

    i = -1
    while(i <= sample_size):
        pass
