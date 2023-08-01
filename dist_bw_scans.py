# source: https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi


def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=8):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    if data_1 is not None:
        x_p, y_p = data_1
        ax.plot(x_p, y_p, color='#336699', marker='o', linestyle=":", label=label_1) # markersize=markersize_1, 
    if data_2 is not None:
        x_q, y_q = data_2
        ax.plot(x_q, y_q, color='orangered', marker='o', linestyle=":", label=label_2) # markersize=markersize_2, 
    ax.legend()
    return ax


def returnFinites(array):
    arr_isfinite = np.all(np.isfinite(array), axis=1)
    return array[arr_isfinite, :]

def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        p_point = P[:, i]
        min_dist = sys.maxsize
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences


def print_distances(P, Q, correspondences):
    sum = 0
    for i in correspondences:
        index_p = i[0]
        index_q = i[1]
        distance = np.linalg.norm(P[:, index_p] - Q[:, index_q])
        sum += distance
        print(P[:, index_p], "-", Q[:, index_q], " = ", distance)
    print("Sum: ", sum)



def main():
    # saveFile = 'data_add/points_diff_scans_infV.npz'
    saveFile = 'data/points_from_sim_ranges_inf.npz'

    # load the two laser scans. they are already in a 4x1 format
    diff_points_from_ranges = np.load(saveFile)
    dspoints_p1_4x1 = returnFinites(diff_points_from_ranges['points_p1'])
    dspoints_p2_4x1 = returnFinites(diff_points_from_ranges['points_p2'])


    true_data = dspoints_p1_4x1[:, 0:2].T
    moved_data = dspoints_p2_4x1[:, 0:2].T

    # Assign to variables we use in formulas.
    Q = true_data
    P = moved_data

    correspondences = get_correspondence_indices(P, Q)
    print_distances(P, Q, correspondences)

if __name__ == '__main__':
    main()

