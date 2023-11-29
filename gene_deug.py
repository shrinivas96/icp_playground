#
# source: https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ranges_and_cartesian import read_scans_write_points, pol2cart

#
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 12
BIGG_SIZE = 14
BIGGER_SIZE = 16

plt.rc('xtick', labelsize=BIG_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIG_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#
# Inverse of a CDF related functions

def RationalApproximation(t):
    # Abramowitz and Stegun formula 26.2.23.
    # The absolute value of the error should be less than 4.5 e-4.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0)

def NormalCDFInverse(p):
    # See article above for explanation of this section.
    if (p < 0.5):
        # F^-1(p) = - G^-1(p)
        return -RationalApproximation( np.sqrt(-2.0*np.log(p)) )
    else:
        # F^-1(p) = G^-1(1-p)
        return RationalApproximation( np.sqrt(-2.0*np.log(1-p)) ) 

#
def plot_data(data_1, data_2, label_1, label_2, markersize_1=5, markersize_2=5):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if data_1 is not None:
        x_p, y_p = data_1
        ax.plot(x_p, y_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label_1)
    if data_2 is not None:
        x_q, y_q = data_2
        ax.plot(x_q, y_q, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label_2)
    ax.legend()
    return ax

#
def plot_data_multiple(datas: list, labels: list, markersizes: list, markersize_2=5):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    colors = ['#336699', 'orangered', "limegreen"]
    for i in range(len(datas)):
        x_p, y_p = datas[i]
        ax.plot(x_p, y_p, color=colors[i], markersize=markersizes[i], marker='o', linestyle=":", label=labels[i])
    ax.legend()
    return ax

#
def returnMetric(sum):
    num_points = 180
    laserRangeVaraince = 0.015933206879414907
    normalVariance = 2*laserRangeVaraince
    muHalfNormal = np.sqrt(2 * normalVariance / np.pi)
    varianceHalfNormal = normalVariance * (1 - (2/np.pi))
    metric = (sum - (muHalfNormal*num_points)) / np.sqrt(varianceHalfNormal*num_points)

    return metric

def returnFinites(array):
    arr_isfinite = np.all(np.isfinite(array), axis=1)
    return array[arr_isfinite, :]

#
def return_ranges(fileToRead):
    # scan file contains just two lines of scan copy pasted from logger on remote machine
    with open(fileToRead, "r") as file:
        ranges_p1 = []
        ranges_p2 = []
        
        # first line
        for x in file.readline().split(", "):
            if(x == "inf"):
                # x = "0.0"
                pass
            ranges_p1.append(float(x.rstrip("\n")))
        
        # second line
        for x in file.readline().split(", "):
            if(x == "inf"):
                # x = "0.0"
                pass
            ranges_p2.append(float(x.rstrip("\n")))
        
        ranges_p1 = np.array(ranges_p1)
        ranges_p2 = np.array(ranges_p2)
    return ranges_p1, ranges_p2

def distortion(scan: np.ndarray, mask_range_ = 0, mask_start_idx_ = 0, window_size_ = 50, unif_min=0.1):
    num_ranges = len(scan)
    for k in range(num_ranges):
        if (k >= mask_start_idx_ and k <= mask_start_idx_+window_size_):
            scan[k] = mask_range_
        scan[k] += np.random.uniform(-unif_min, unif_min)
    return scan

def convert_to_cartesian(ranges_p1:np.ndarray):

    v_cart = np.vectorize(pol2cart)

    # corresponding angles in [0, 2pi) for the range scans
    angles = np.arange(0, 360) * np.pi/180
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

#
readFile = "data_add/logRealSimScanComparision.dat"
saveFile = 'writing_data/pointsSimDistortReal.npz'

ranges_p1_real, ranges_p2_sim = return_ranges(readFile)

points_p1_real_4x1 = convert_to_cartesian(np.copy(ranges_p1_real))
points_p2_sim_4x1 = convert_to_cartesian(ranges_p2_sim)

ranges_p1_dist = distortion(ranges_p1_real, mask_range_=1.5, mask_start_idx_=-1, window_size_=0, unif_min=0.05)
points_p1_dist_4x1 = convert_to_cartesian(ranges_p1_dist)


dspoints_p1_4x1 = returnFinites(points_p1_dist_4x1)
dspoints_p2_4x1 = returnFinites(points_p2_sim_4x1)

true_data = dspoints_p1_4x1[:, 0:2].T
moved_data = dspoints_p2_4x1[:, 0:2].T

#
ranges_p1_distm = distortion(ranges_p1_real, mask_range_=1.5, mask_start_idx_=-1, window_size_=0, unif_min=0.08)
points_p1_distm_4x1 = convert_to_cartesian(ranges_p1_distm)
points_p1_distm_4x1 = returnFinites(points_p1_distm_4x1)

#
ax = plot_data_multiple([points_p1_real_4x1[:, 0:2].T, true_data, points_p1_distm_4x1[:, 0:2].T],
               labels=['Scan at $(x, y, \\theta)$', 'Distortion $\\in \\mathcal{U}_{[-0.05, 0.05]}$', 'Distortion $\\in \\mathcal{U}_{[-0.08, 0.08]}$'],
               markersizes=[4, 4, 4])

plt.grid(True)
plt.legend()
plt.show()

#
ax = plot_data(true_data, moved_data, 
               label_1='Real Scan', label_2='Simulated Scan',
               markersize_1=6, markersize_2=6)

plt.grid(True)
plt.legend()
plt.show()

#
# create an initial transformation for ICP to find the original
beta = 0
cos_a = np.cos(beta)
sin_a = np.sin(beta)

fake_trans = np.eye(4)
fake_trans[0:2, 0:2] = np.array([[cos_a, -sin_a],       # rotation matrix
                                 [sin_a, cos_a]])
fake_trans[0, 3] = -0.03                                 # x translation
# fake_trans[1, 3] = 0.3                                  # y translation

#
# create a point cloud object for open3d and input your points in there
source_pcd = o3d.geometry.PointCloud()
target_pcd = o3d.geometry.PointCloud()
icp_aligned_pcd = o3d.geometry.PointCloud()

source_pcd.points = o3d.utility.Vector3dVector(dspoints_p1_4x1[:, 0:3])
target_pcd.points = o3d.utility.Vector3dVector(dspoints_p2_4x1[:, 0:3])

threshold = 0.02*1e1

#
# do icp and get the resulting transformation. then transform points according to new transformation

# performing the ICP with open3d
reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, fake_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

# the final trasnformation as a 4x4 hom. trans.
T_icp = reg_p2p.transformation

# apply transformation to points of p1
points_p1_reg_icp = np.matmul(dspoints_p1_4x1, T_icp.T)
icp_aligned_pcd.points = o3d.utility.Vector3dVector(points_p1_reg_icp[:, 0:3])

print(reg_p2p)
print("Transformation is:")
print(T_icp)
# draw_registration_result(source_pcd, target_pcd, reg_p2p.transformation)

#
ax = plot_data(moved_data, points_p1_reg_icp[:, 0:2].T, label_1='Simulated Scan', label_2='Real Scan after ICP')

plt.grid(True)
plt.legend()
plt.show()

#
point_clouds_aligned = "writing_data/pointsSimAlignedReal_2023-06-12-16-36.npz"
np.savez(point_clouds_aligned, points_p1=dspoints_p2_4x1, points_p2=points_p1_reg_icp)

#
# computing corresponding distance between the two aligned scans
dist_bw_source_target = np.array(source_pcd.compute_point_cloud_distance(target_pcd))
dist_bw_source_target_rw = np.array(target_pcd.compute_point_cloud_distance(source_pcd))
print("Before alignment: ", dist_bw_source_target.sum(), dist_bw_source_target_rw.sum())

dist_bw_target_alined = np.array(target_pcd.compute_point_cloud_distance(icp_aligned_pcd))
dist_bw_target_alined_rw = np.array(icp_aligned_pcd.compute_point_cloud_distance(target_pcd))
print("After alignment: ", dist_bw_target_alined.sum(), dist_bw_target_alined_rw.sum())

#
sum_st = dist_bw_target_alined.sum()
sum_st_rw = dist_bw_target_alined_rw.sum()
print("Sum of the distances: ", sum_st, sum_st_rw)

#
alpha = 0.1
oneMinusAlpha = 1 - alpha
f_inv = NormalCDFInverse(oneMinusAlpha)

print("Metric: ", returnMetric(sum_st))
print("Metric: ", returnMetric(sum_st_rw))
print("F^-1(1-alpha): ", f_inv)
# print(NormalCDFInverse(alpha))

#



