import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ranges_and_cartesian import read_scans_write_points, pol2cart

# Inverse of a CDF related functions


def RationalApproximation(t):
    # Abramowitz and Stegun formula 26.2.23.
    # The absolute value of the error should be less than 4.5 e-4.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - ((c[2] * t + c[1]) * t + c[0]) / (
        ((d[2] * t + d[1]) * t + d[0]) * t + 1.0
    )


def NormalCDFInverse(p):
    # See article above for explanation of this section.
    if p < 0.5:
        # F^-1(p) = - G^-1(p)
        return -RationalApproximation(np.sqrt(-2.0 * np.log(p)))
    else:
        # F^-1(p) = G^-1(1-p)
        return RationalApproximation(np.sqrt(-2.0 * np.log(1 - p)))


def plot_data(data_1, data_2, label_1, label_2, markersize_1=5, markersize_2=5, figSize=(12, 7)):
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(111)
    ax.axis("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if data_1 is not None:
        x_p, y_p = data_1
        ax.plot(
            x_p,
            y_p,
            color="#336699",
            markersize=markersize_1,
            marker="o",
            linestyle=":",
            label=label_1,
        )
    if data_2 is not None:
        x_q, y_q = data_2
        ax.plot(
            x_q,
            y_q,
            color="orangered",
            markersize=markersize_2,
            marker="o",
            linestyle=":",
            label=label_2,
        )
    ax.legend()
    return ax


def plot_data_multiple(datas: list, labels: list, markersizes: list, markersize_2=5):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    colors = ["#336699", "orangered", "forestgreen"]
    for i in range(len(datas)):
        x_p, y_p = datas[i]
        ax.plot(
            x_p,
            y_p,
            color=colors[i],
            markersize=markersizes[i],
            marker="o",
            linestyle=":",
            label=labels[i],
        )
    ax.legend()
    return ax


def returnMetric(sum):
    num_points = 180
    laserRangeVaraince = 0.015933206879414907
    normalVariance = 2 * laserRangeVaraince
    muHalfNormal = np.sqrt(2 * normalVariance / np.pi)
    varianceHalfNormal = normalVariance * (1 - (2 / np.pi))
    metric = (sum - (muHalfNormal * num_points)) / np.sqrt(
        varianceHalfNormal * num_points
    )

    return metric


def returnFinites(array):
    arr_isfinite = np.all(np.isfinite(array), axis=1)
    return array[arr_isfinite, :]


def return_ranges(fileToRead):
    # scan file contains just two lines of scan copy pasted from logger on remote machine
    with open(fileToRead, "r") as file:
        ranges_p1 = []
        ranges_p2 = []

        # first line
        for x in file.readline().split(", "):
            if x == "inf":
                # x = "0.0"
                pass
            ranges_p1.append(float(x.rstrip("\n")))

        # second line
        for x in file.readline().split(", "):
            if x == "inf":
                # x = "0.0"
                pass
            ranges_p2.append(float(x.rstrip("\n")))

        ranges_p1 = np.array(ranges_p1)
        ranges_p2 = np.array(ranges_p2)
    return ranges_p1, ranges_p2


def distortion(
    scan: np.ndarray, mask_range_=0, mask_start_idx_=0, window_size_=50, unif_min=0.1
):
    num_ranges = len(scan)
    for k in range(num_ranges):
        if k >= mask_start_idx_ and k <= mask_start_idx_ + window_size_:
            scan[k] = mask_range_
        scan[k] += np.random.uniform(-unif_min, unif_min)
    return scan


def convert_to_cartesian(ranges_p1: np.ndarray):
    v_cart = np.vectorize(pol2cart)

    # corresponding angles in [0, 2pi) for the range scans
    angles = np.arange(0, 360) * np.pi / 180
    angles = np.reshape(angles, (360, 1))

    ranges_p1 = np.reshape(ranges_p1, (360, 1))

    # ones and zeros array
    zeros = np.zeros((360, 1))
    ones = np.ones((360, 1))

    # converting polar to cartesian
    xy_p1 = v_cart(ranges_p1, angles)

    # appending 0's and 1's to make it usable with 4x4 matrices.
    # 0's in third dimension just means it is one plane in 3D points
    points_p1 = np.column_stack((xy_p1[0], xy_p1[1], zeros, ones))
    return points_p1
